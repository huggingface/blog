---
title: "Open-source LLMs as LangChain Agents" 
thumbnail: /blog/assets/131_aws-partnership/aws-partnership-thumbnail.png
authors:
- user: m-ric
- user: Jofthomas
- user: andrewrreed
---
# Open-source LLMs as LangChain Agents

## Introduction

Large Language Models (LLMs) trained for [causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) can tackle a wide range of tasks, but they often struggle with basic tasks like logic, calculation, and search. The worst scenario is when they perform poorly in a domain, such as math, yet still attempt to handle all the calculations themselves.

To overcome this weakness, amongst other approaches, one can integrate the LLM into a system where it can call tools: such a system is called an LLM Agent.  

## Table of Contents

- [What are agents?](#what-are-agents)
- [Running agents with LangChain](#running-agents-with-langchain)
- [Agents Showdown: how do different LLMs perform as general purpose reasoning agents?](#agents-showdown-how-do-different-llms-perform-as-general-purpose-reasoning-agents)
    - [Evaluation dataset](#evaluation-dataset)
    - [Results](#results)


## What are agents?

The definition of LLM Agents is quite broad: LLM agents are all systems that use LLMs as their engine and can perform actions on their environment based on observations. They can use several iterations of the Perception ⇒ Reflexion ⇒ Action cycle to achieve their task and are often augmented with planning or knowledge management systems to enhance their performance. You can find a good review of the Agents landscape in [Xi et al., 2023](https://huggingface.co/papers/2309.07864).

Today, we are focusing on **ReAct agents**. [ReAct](https://huggingface.co/papers/2210.03629) is an approch to building agents based on the concatenation of two words, "**Reasoning**" and "**Acting**." In the prompt, we describe the model, which tools it can use, and ask it to think “step by step” (also called [Chain-of-Thought](https://huggingface.co/papers/2201.11903) behavior) to plan and execute its next actions to reach the final answer. 

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png" alt="drawing" width=90%>
</p>

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

Generally, the difficult parts of running an Agent system for the LLM engine are:

1. From supplied tools, choose the one that will help advance to a desired goal: e.g. when asked `"What is the smallest prime number greater than 30,000?"`, the agent could call the `Search` tool with `"What is he height of K2"` but it won't help.
2. Call tools with a rigorous argument formatting: for instance when trying to calculate the speed of a car that went 3 km in 10 minutes, you have to call tool `Calculator` to divide `distance` by `time` : even if your Calculator tool accepts calls in the JSON format: `{”tool”: “Calculator”, “args”: “3km/10min”}` , there are many pitfalls, for instance:
    - Misspelling the tool name: `“calculator”` or `“Compute”` wouldn’t work
    - Giving the name of the arguments instead of their values: `“args”: “distance/time”`
    - Non-standardized formatting: `“args": "3km in 10minutes”`
3. Efficiently ingesting and using the information gathered in the past observations, be it the initial context or the observations returned after using tool uses.

With that in mind, let us get a low level understanding of how agents are able to use tools!

Take a look at [this notebook](https://colab.research.google.com/drive/1j_vsc28FwZEDocDxVxWJ6Fvxd18FK8Gl?usp=sharing): we implement a barebones tool call example with the Transformers library. 

So, how would it look like in a complete Agent setup?

## Running agents with LangChain

We’ve just integrated a `ChatHuggingFace` wrapper that lets you create agents based on open-source models in [🦜🔗LangChain](https://www.langchain.com/).

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

## Agents Showdown: how do different LLMs perform as general purpose reasoning agents?


To understand how open-source LLMs perform as general purpose reasoning agents, we have evaluated strong models on questions requiring using a few basic tools.

### Evaluation dataset

We selected questions that can be answered using basic tools: a simple calculator and access to internet search.

- For Internet search capability: we have selected questions from [HotpotQA](https://huggingface.co/datasets/hotpot_qa): this is originally a retrieval dataset, but it can be used for general question answering, with access to the internet. Some questions originally need to combine information from various sources: in our setting, this means performing several steps of internet search to combine the results.
- For testing calculator usage, we added questions from [GSM8K](https://huggingface.co/datasets/gsm8k): this dataset tests grade-school math ability, and is entirely solvable by correctly leveraging the 4 operators (add, subtract, multiply, divide).
- We also picked questions from [GAIA](https://huggingface.co/papers/2311.12983), a very difficult benchmark for General AI Assistants. The questions in the original dataset can require many other different tools, such as a code interpreter or pdf reader: we hand-picked questions that do not require anything except search and calculator.

Evaluation was performed with GPT4-as-a-judge using a prompt based on the [Prometheus prompt format](https://huggingface.co/kaist-ai/prometheus-13b-v1.0), giving results on the Likert Scale: see the exact prompt [here](https://github.com/A-Roucher/benchmark_agents/blob/master/scripts/prompts.py).

### Models

On the open-source side, we evaluate:
- [Llama2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
- [Zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

These open-source models are evaluated in LangChain's [ReAct implementation](https://github.com/langchain-ai/langchain/tree/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/langchain/langchain/agents/react)
For the open-source models, as per the standard LangChain ReAct prompt, we prompt them to output their function calls in this common format:

```json
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}
```

For comparison, we have also evaluated GPT3.5 and GPT4 on the same examples using LangChain's [OpenAI specific function calling agent](https://github.com/langchain-ai/langchain/tree/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/langchain/langchain/agents/openai_functions_agent).
Since this uses their own function calling template, on which they were fine-tuned, this means the OpenAI models should be at their top performance.

### Results

Since the open-source models were not specifically fine-tuned for calling functions in the given output format, they are at a slight disadvantage to the OpenAI agents.

Despite this, some models perform really well!

Here’s an example of Mixtral-8x7B answering the question:

> “**Which city has a larger population, Guiyang or Tacheng?”**
> 

```bash
    **Thought:** To answer this question, I need to find the current populations of both Guiyang and Tacheng. I will use the search tool to find this information.
    **Action:**

    ```json
        {
        "action": "search",
        "action_input": "current population of Guiyang"
        }
    ```
    4.697 million (2016)

    **Thought:** I have obtained the population of Guiyang, which is 4.697 million (as of 2016). Now, I will use the search tool again to find the population of Tacheng.
    **Action:**
    ```json
        {
        "action": "search",
        "action_input": "current population of Tacheng"
        }
    ```
    Population: Xinjiang: Tacheng data was reported at 1,110.000 Person in 2021. This records a decrease from the previous number of 1,138.638 Person for 2020.
    Population: Xinjiang: Tacheng data is updated yearly, averaging 1,023.000 Person from Dec 2005 to 2021, with 17 observations.

    I have obtained the population of Tacheng, which is approximately 1.11 million (as of 2021). Comparing the two populations, Guiyang has a larger population than Tacheng.

    **Thought:** I now know the final answer
    **Final Answer:** Guiyang has a larger population, which is approximately 4.697 million (as of 2016), compared to Tacheng's population of approximately 1.11 million (as of 2021).
```

Here is a benchmark of the models on our evaluation dataset:

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/benchmark_agents.png" alt="benchmark of agents performance" width="90%">
</p>

As you can see, some open-source models perform poorly: while this was expected for the small Zephyr-7b, Llama2-70b performs surprisingly poorly.

But Mixtral-8x7B holds its own really well compared to other models: it performs nearly equivalent to GPT3.5. It is the best of the OS models we tested to power Agent workflows! 🏆

This is out-of-the-box performance: __contrary to GPT3.5, Mixtral was not finetuned for agent workflows__ (to our knowledge), which somewhat hinders its performance. For instance, on GAIA, 10% of questions fail because Mixtral tries to call a tool with incorrectly formatted arguments. **With proper finetuning for the function calling and task planning skills, Mixtral’s score would likely be even higher.** We strongly recommend open-source builders to start finetuning Mixtral for agents, to surpass the next challenger: GPT4! 🚀

**Closing remarks:**

- The GAIA benchmark, although it is tried here on a small subsample of questions and a few tools, seems like a very good indicator of overall model performance, since it often involves several reasoning steps and rigorous logic.
- The agent workflows allow LLMs to increase performance: for instance, on GSM8K, [GPT4’s technical report](https://arxiv.org/pdf/2303.08774.pdf) reports 92% for 5-shot CoT prompting: giving it a calculator allows us to reach 95% in zero-shot . For Mixtral-8x7B, the [LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) reports 57.6% with 5-shot, we get 73% in zero-shot. (keep in mind that we tested only a subset of 20 questions)