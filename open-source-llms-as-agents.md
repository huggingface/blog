---
title: "Open-source LLMs as LangChain Agents" 
thumbnail: /blog/assets/131_aws-partnership/aws-partnership-thumbnail.png
authors:
- user: m-ric
- user: Jofthomas
- user: andrewrreed
---
# Introduction

Large Language Models (LLMs) trained to perform [causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) can tackle a wide range of tasks, but they often struggle with basic tasks like logic, calculation, and search. The worst scenario is when they perform poorly in a domain, such as math, yet still attempt to handle all the calculations themselves.

One approach to overcome this weakness is to embed the LLM into a system where it has the ability to call tools: such a system is called an LLM Agent.  

The definition of LLM Agents is quite broad: all systems that use LLMs as their engine, and have the possibility to perform actions on their environment based on observations. They can use several iterations of the Perception ⇒ Reflexion ⇒ Action cycle to achieve their task, and are often augmented with planning or knowledge management systems to enhance their performance. You can find a good review of the Agents landscape in [Xi et al., 2023](https://huggingface.co/papers/2309.07864).

Today, we are focusing on `ReAct agents`. ReAct is the concatenation of two words, “**Reasoning**” and “**Acting**”. In the prompt, we describe the model which tools it can use, and ask it to think “step by step” (also called `Chain-of-thought` behaviour) to plan and execute its next actions in order to reach the final answer. 

![Sans-titre-2024-01-10-2238.png](%5BDRAFT%5D%20Open-source%20LLMs%20as%20LangChain%20Agents%20632cb4cb4e764465a490eec01a7a6d95/Sans-titre-2024-01-10-2238.png)

This graph seems very high level, but under the hood it’s quite simple: the LLM is called in a loop with a prompt containing in essence:

```
Here is a question: {question}. 
You have access to these tools: {tools_descriptions}. 
You should first reflect with ‘Thought: {your_thoughts}’, then you either:
- call a tool with the proper JSON formatting,
- or your print your final answer starting with the prefix ‘Final Answer:’
```

Then you parse the LLM’s output:

- if it contains the string `‘Final Answer:’`, the loop ends and you print the answer
- else, the LLM should have output a tool call: you can parse this output to get the tool name and arguments, then call said tool with said arguments. Then the output of this tool call is appended to the prompt, and you call the LLM again with this extended information, until it has enough information to finally provide a final answer to the question.

Generally, the difficult parts of running an Agent system for the LLM engine is to:

1. Choose the right tool calls to really progress towards its goal
2. Call tools with a rigorous argument formatting: for instance when trying to calculate the speed of a car that went 3 km in 10 minutes, you have to call tool `Calculator` to divide `distance` by `time` : even if your Calculator tool accepts calls in the JSON format: `{”tool”: “Calculator”, “args”: “3km/10min”}` , there are many pitfalls:
    - Misspelling the tool name: `“calculator”` or `“Compute”` wouldn’t work
    - Giving the name of the arguments instead of their values: `“args”: “distance/time”`
    - Non-standardized formatting: `“args": "3km in 10minutes”`
    - …
3. Efficiently ingesting and using the information gathered in the past observations, be it the initial context or the observations returned after using tool uses.

With that in mind, let us get a low level understanding of how agents are able to use tools! We are going to implement a barebones tool call example with the Transformers library. 

**[@Joffrey THOMAS ‘s notebook]**

So, how would it look like in a complete Agent setup?

### Running agents with [🦜🔗LangChain](https://www.langchain.com/)

We’ve just integrated a `ChatHuggingFace` wrapper which will let you create agents based on open-source models in LangChain.

The code to create the ChatModel and give it tools is really simple, you can check it all in the [Langchain doc](https://python.langchain.com/docs/integrations/chat/huggingface). 

```python
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

chat_model = ChatHuggingFace(llm=llm)
```

You can make the chat_model into an Agent by giving it a ReAct style prompt and tools:

```python
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper

# setup tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "Who is the current holder of the speedskating world record? What is her current age raised to the 0.43 power?"
    }
)
```

# Agents Showdown: how do different LLMs perform as general purpose reasoning agents?

To understand how open-source LLM’s perform as general purpose reasoning agents, we evaluated four strong models (`Llama2-70b-chat`, `Mixtral-8x7B-Instruct-v0.1`, `OpenHermes-2.5-Mistral-7B`, `Zephyr-7b-beta`) in a [ReACT workflow](https://github.com/langchain-ai/langchain/tree/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/langchain/langchain/agents/react) where they were tasked with answer questions that require basic tool usage. We also evaluated GPT3.5 and GPT4 on the same examples using the [OpenAI specific function calling agent](https://github.com/langchain-ai/langchain/tree/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/langchain/langchain/agents/openai_functions_agent) for comparison.

### Evaluation Dataset

We selected questions that can be answered using basic tools: a simple calculator and access to internet search.

- For Internet search capability: we selected questions from [HotpotQA](https://huggingface.co/datasets/hotpot_qa): this is originally a retrieval dataset, but it can be used for general question answering, with access to the internet. Some questions originally need to combine information from various sources: in our setting, this means performing several steps of internet search to combine the results.
- For testing calculator usage, we add questions from [GSM8K](https://huggingface.co/datasets/gsm8k): this dataset tests grade-school math ability, and is entirely solvable by correctly leveraging the 4 operators (add, subtract, multiply, divide).
- We also picked questions from [GAIA](https://huggingface.co/papers/2311.12983), a very difficult benchmark for General AI Assistants. The questions in the original dataset can require many other different tools, such as a code interpreter or pdf reader: we hand-picked questions that do not require anything except search and calculator.

Evaluation was performed with GPT4-as-a-judge using [the prompt format referenced here](https://github.com/A-Roucher/benchmark_agents/blob/master/prompts.py).

### Results

OpenAI models were evaluated in their own function calling template with which they were fine-tuned. For the other, open-source models, as per the standard LangChain ReAct prompt, we prompt them to output their function calls in this common format:

```python
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Since these open-source models were not specifically tuned for calling functions in this output format, they are at a slight disadvantage to the OpenAI agents. Despite this, some models perform really well!

Here’s an example of Mixtral-8x7B answering the question:

> “**Which city has a larger population, Guiyang or Tacheng?”**
> 

```python
***Thought:** To answer this question, I need to find the current populations of both Guiyang and Tacheng. I will use the search tool to find this information.
**Action:**
```json
{
  "action": "search",
  "action_input": "current population of Guiyang"
}
```
4.697 million (2016)*

***Thought:*** *I have obtained the population of Guiyang, which is 4.697 million (as of 2016). Now, I will use the search tool again to find the population of Tacheng.
**Action:**
```json
{
  "action": "search",
  "action_input": "current population of Tacheng"
}
```
Population: Xinjiang: Tacheng data was reported at 1,110.000 Person th in 2021. This records a decrease from the previous number of 1,138.638 Person th for 2020.
Population: Xinjiang: Tacheng data is updated yearly, averaging 1,023.000 Person th from Dec 2005 to 2021, with 17 observations.

I have obtained the population of Tacheng, which is approximately 1.11 million (as of 2021). Comparing the two populations, Guiyang has a larger population than Tacheng.

**Thought:** I now know the final answer
**Final Answer:** Guiyang has a larger population, which is approximately 4.697 million (as of 2016), compared to Tacheng's population of approximately 1.11 million (as of 2021).*
```

Here is a benchmark of the models on our evaluation dataset:

![Untitled](%5BDRAFT%5D%20Open-source%20LLMs%20as%20LangChain%20Agents%20632cb4cb4e764465a490eec01a7a6d95/Untitled.png)

As you can see, some open-source models perform poorly: while this was expected for the small Zephyr-7b, Llama2-70b performs surprisingly poorly.

But Mixtral-8x7B holds its own really well compared to other models: it has a performance nearly equivalent to GPT3.5. It is the best of the OS models we tested to power Agent workflows! 🏆

And this is out-of-the-box performance: with proper finetuning for the function calling and task planning skills, which would get rid of some parsing errors hindering performance, its score could be pushed even higher. For instance on GSM8K, 10% of calls fail due to Mixtral trying to call a tool with poorly formatted arguments: if these were corrected by finetuning, assuming the same success ratio as for other questions (~70%), the agent would get +7% score increase on GSM8K.

This performance is really promising: thus we strongly recommend open-source builders to start finetuning Mixtral for agents! 🚀

The GAIA benchmark, although it is tried here on a small subsample of questions and a few tools, seems like a very good indicator of overall model performance, since it generally involves several reasoning steps and rigorous logic.