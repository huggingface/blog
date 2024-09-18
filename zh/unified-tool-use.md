---
title: "对 LLM 工具使用进行统一"
thumbnail: /blog/assets/unified-tool-use/thumbnail.png
authors:
- user: rocketknight1
translators:
- user: MatrixYao
---

# 对 LLM 工具使用进行统一

我们为 LLM 确立了一个跨模型的**统一工具调用 API**。有了它，你就可以在不同的模型上使用相同的代码，在 [Mistral](https://huggingface.co/mistralai)、[Cohere](https://huggingface.co/CohereForAI)、[NousResearch](https://huggingface.co/NousResearch) 或 [Llama](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f) 等模型间自由切换，而无需或很少需要根据模型更改工具调用相关的代码。此外，我们还在 `transformers` 中新增了一些实用接口以使工具调用更丝滑，我们还为此配备了[完整的文档](https://huggingface.co/docs/transformers/main/chat_templat#advanced-tool-use--function-calling)以及端到端工具使用的[示例](https://github.com/huggingface/blog/blob/main/notebooks/unified-tool-calling.ipynb)。我们会持续添加更多的模型支持。


## 引言

LLM 工具使用这个功能很有意思 —— 每个人都认为它很棒，但大多数人从未亲测过。它的概念很简单：你给 LLM 提供一些工具（即：可调用的函数），LLM 在响应用户的查询的过程中可自主判断、自行调用它们。比方说，你给它一个计算器，这样它就不必依赖其自身不靠谱的算术能力；你还可以让它上网搜索或查看你的日历，或者授予它访问公司数据库的权限（只读！），以便它可以提取相应信息或搜索技术文档。

工具调用使得 LLM 可以突破许多自身的核心限制。很多 LLM 口齿伶俐、健谈，但涉及到计算和事实时往往不够精确，并且对小众话题的具体细节不甚了解。它们还不知道训练数据截止日期之后发生的任何事情。它们是通才，但除了你在系统消息中提供的信息之外，它们在开始聊天时对你或聊天背景一无所知。工具使它们能够获取结构化的、专门的、相关的、最新的信息，这些信息可以帮助其成为真正有帮助的合作伙伴，而不仅仅是令人着迷的新奇玩意儿。

然而，当你开始真正尝试工具使用时，问题出现了！文档很少且互相之间不一致，甚至矛盾 —— 对于闭源 API 和开放模型无不如此！尽管工具使用在理论上很简单，但在实践中却常常成为一场噩梦：如何将工具传递给模型？如何确保工具提示与其训练时使用的格式相匹配？当模型调用工具时，如何将其合并到聊天提示中？如果你曾尝试过动手实现工具使用，你可能会发现这些问题出奇棘手，而且很多时候文档并不完善，有时甚至会帮倒忙。

更糟糕的是，不同模型的工具使用的实现可能迥异。即使在定义可用工具集这件最基本的事情上，一些模型厂商用的是 JSON 模式，而另一些模型厂商则希望用 Python 函数头。即使那些希望使用 JSON 模式的人，细节上也常常会有所不同，因此造成了巨大的 API 不兼容性。看！用户被摁在地板上疯狂摩擦，同时内心困惑不已。

为此，我们能做些什么呢？

## 聊天模板

Hugging Face 时空的忠​​粉会记得，开源社区过去在**聊天模型**方面也面临过类似的挑战。聊天模型使用 `<|start_of_user_turn|>` 或 `<|end_of_message|>` 等控制词元来让模型知道聊天中发生了什么，但不同的模型训练时使用的控制词元完全不同，这意味着用户需要为他们用的模型分别编写特定的格式化代码。这在当时是一个非常头疼的问题。

最终的解决方案是**聊天模板** - 即，模型会自带一个小小的 [Jinja](https://jinja.palletsprojects.com/en/3.1.x/) 模板，它能用正确的格式来规范每个模型的聊天格式和控制词元。聊天模板意味着用户能用通用的、与模型无关的方式编写聊天，并信任 Jinja 模板来处理模型格式相关的事宜。

基于此，支持工具使用的一个显而易见的方法就是扩展聊天模板的功能以支持工具。这正是我们所做的，但工具给模板方案带来了许多新的挑战。我们来看看这些挑战以及我们是如何解决它们的吧。希望在此过程中，你能够更深入地了解该方案的工作原理以及如何更好利用它。

## 将工具传给聊天模板

在设计工具使用 API 时，首要需求是定义工具并将其传递给聊天模板的方式应该直观。我们发现大多数用户的流程是：首先编写工具函数，然后弄清楚如何据其生成工具定义并将其传递给模型。一个自然而然的想法是：如果用户可以简单地将函数直接传给聊天模板并让它为他们生成工具定义那就好了。

但问题来了，“传函数”的方式与使用的编程语言极度相关，很多人是通过 [JavaScript](https://huggingface.co/docs/transformers.js/en/index) 或 [Rust](https://huggingface.co/docs/text-generation-inference/en/index) 而不是 Python 与聊天模型交互的。因此，我们找到了一个折衷方案，我们认为它可以两全其美：**聊天模板将工具定义为 JSON 格式，但如果你传 Python 函数给模板，我们会将其自动转换为 JSON 格式.** 这就产生了一个漂亮、干净的 API：

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
    tools=tools,
    add_generation_prompt=True,
    return_tensors="pt"
)
```

在 `apply_chat_template` 内部，`get_current_temperature` 函数会被转换成完整的 JSON 格式。想查看生成的格式，可以调用 `get_json_schema` 接口：

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
如果你更喜欢手动控制或者使用 Python 以外的语言进行编码，则可以将工具组织成 JSON 格式直接传给模板。但是，当你使用 Python 时，你可以无需直接处理 JSON 格式。你仅需使用清晰的**函数名、**准确的**类型提示**以及完整的含**参数文档字符串**的**文档字符串**来定义你的工具函数，所有这些都将用于生成模板所需的 JSON 格式。其实，这些要求本来就已是 Python 最佳实践，你本应遵守，如果你之前已经遵守了，那么无需更多额外的工作，你的函数已经可以用作工具了！

请记住：无论是从文档字符串和类型提示生成还是手动生成，JSON 格式的准确性，对于模型了解如何使用工具都至关重要。模型永远不会看到该函数的实现代码，只会看到 JSON 格式，因此它们越清晰、越准确越好！

## 在聊天中调用工具

用户（以及模型文档😬）经常忽略的一个细节是，当模型调用工具时，实际上需要将**两条**消息添加到聊天历史记录中。第一条消息是模型**调用**工具的信息，第二条消息是**工具的响应**，即被调用函数的输出。 

工具调用和工具响应都是必要的 - 请记住，模型只知道聊天历史记录中的内容，如果它看不到它所作的调用以及传递的参数，它将无法理解工具的响应。`22` 本身并没有提供太多信息，但如果模型知道它前面的消息是 `get_current_temperature("Paris, France")`，则会非常有帮助。

不同模型厂商对此的处理方式迥异，而我们将工具调用标准化为**聊天消息中的一个域**，如下所示：

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

## 在聊天中添加工具响应

工具响应要简单得多，尤其是当工具仅返回单个字符串或数字时。

```python
message = {
    "role": "tool", 
    "name": "get_current_temperature", 
    "content": "22.0"
}
chat.append(message)
```

## 实操

我们把上述代码串联起来搭建一个完整的工具使用示例。如果你想在自己的项目中使用工具，我们建议你尝试一下我们的代码 - 尝试自己运行它，添加或删除工具，换个模型并调整细节以感受整个系统。当需要在软件中实现工具使用时，这种熟悉会让事情变得更加容易！为了让它更容易，我们还提供了这个示例的 [notebook](https://github.com/huggingface/blog/blob/main/notebooks/unified-tool-calling.ipynb)。

首先是设置模型，我们使用 `Hermes-2-Pro-Llama-3-8B`，因为它尺寸小、功能强大、自由使用，且支持工具调用。但也别忘了，更大的模型，可能会在复杂任务上获得更好的结果！

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
```

接下来，我们设置要使用的工具及聊天消息。我们继续使用上文的 `get_current_Temperature`：

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

模型可用工具设定完后，就需要模型生成对用户查询的响应：

```python
out = model.generate(**tool_prompt, max_new_tokens=128)
generated_text = out[0, tool_prompt['input_ids'].shape[1]:]

print(tokenizer.decode(generated_text))
```

我们得到：

```python
<tool_call>
{"arguments": {"location": "Paris, France"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

模型请求使用一个工具！请注意它正确推断出应该传递参数 “Paris, France” 而不仅仅是 “Paris”，这是因为它遵循了函数文档字符串推荐的格式。 

但模型并没有真正以编程方式调用这些工具，就像所有语言模型一样，它只是生成文本。作为程序员，你需要接受模型的请求并调用该函数。首先，我们将模型的工具请求添加到聊天中。

请注意，此步骤可能需要一些手动处理 - 尽管你应始终按照以下格式将请求添加到聊天中，但模型调用工具的请求文本（如 `<tool_call>` 标签）在不同模型之间可能有所不同。通常，它非常直观，但请记住，在你自己的代码中尝试此操作时，你可能需要一些特定于模型的 `json.loads()` 或 `re.search()`！

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

现在，我们真正在 Python 代码中调用该工具，并将其响应添加到聊天中：

```python
message = {
    "role": "tool", 
    "name": "get_current_temperature", 
    "content": "22.0"
}
chat.append(message)
```

然后，就像之前所做的那样，我们按格式更新聊天信息并将其传给模型，以便它可以在对话中使用工具响应：

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

最后，我们得到对用户的最终响应，该响应是基于中间工具调用步骤中获得的信息构建的：

```html
The current temperature in Paris is 22.0 degrees Celsius. Enjoy your day!<|im_end|>
```

## 令人遗憾的响应格式不统一

在上面的例子中，你可能已经发现，尽管聊天模板可以帮助隐藏模型之间在聊天格式以及工具定义格式上的差异，但它仍有未尽之处。当模型发出工具调用请求时，其用的还是自己的格式，因此需要你手动解析它，然后才能以通用格式将其添加到聊天中。值得庆幸的是，大多数格式都非常直观，因此应该仅需几行 `json.loads()`，最坏情况下估计也就是一个简单的 `re.search()` 就可以创建你需要的工具调用字典。

尽管如此，这是最后遗留下来的“不统一”尾巴。我们对如何解决这个问题有一些想法，但尚未成熟，“撸起袖子加油干”吧！

## 总结

尽管还留了一点小尾巴，但我们认为相比以前，情况已经有了很大的改进，之前的工具调用方式分散、混乱且记录不足。我们希望我们为统一作的努力可以让开源开发人员更轻松地在他们的项目中使用工具，以通过一系列令人惊叹的新工具来增强强大的 LLM。从 [Hermes-2-Pro-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B) 等较小模型到 [Mistral-Large](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407)、[Command-R-Plus](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 或[Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct) 等最先进的巨型庞然大物，越来越多的前沿 LLM 已经支持工具使用。我们认为工具将成为下一波 LLM 产品不可或缺的一部分，我们希望我们做的这些改进能让你更轻松地在自己的项目中使用它们。祝你好运！

> 英文原文: <url> https://huggingface.co/blog/unified-tool-use </url>
> 原文作者：Matthew Carrigan
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。