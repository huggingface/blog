---
title: "License to Call: Releasing Transformers Agents v2.0"
thumbnail: /blog/assets/agents/thumbnail.png
authors:
  - user: m-ric
  - user: lysandre
---

# License to Call: Releasing Transformers Agents v2.0

### TL;DR

We are releasing Transformers Agents 2.0!

⇒ 🎁 We introduce two new agent types that **can iterate based on past observations to solve complex tasks**.

⇒ 💡 We aim for the code to be **simple and clear, and for common attributes like the final prompt and tools to be transparent**.

⇒ 🤝 We add **sharing options** to boost community agents.

⇒ 💪 **Extremely performant new agent framework**, allowing a Llama3-70B agent to outperform GPT-4 based agents in the GAIA Leaderboard!

🚀 Go try it out and climb ever higher on the GAIA leaderboard!

## Table of Contents

- [Introduction](#introduction)
- [Our approach](#our-approach)
    - [Main elements](#main-elements)
- [Example use-cases](#example-use-cases)
    - [Self-correcting Retrieval-Augmented-Generation](#self-correcting-retrieval-augmented-generation)
    - [Using a simple multi-agent setup 🤝 for efficient web browsing](#using-a-simple-multi-agent-setup-for-efficient-web-browsing)
- [Testing our agents](#testing-our-agents)
    - [Comparing LLM engines](#comparing-llm-engines)
    - [Climbing up the GAIA Leaderboard with a multi-modal agent](#climbing-up-the-gaia-leaderboard-with-a-multi-modal-agent)
- [Conclusion](#conclusion)

## Introduction

What is an agent?

Large Language Models (LLMs) can tackle a wide range of tasks, but they often struggle with specific tasks like logic, calculation, and search: thus when prompted in domains in which they do not perform well, they often fail to generate a correct answer.

One approach to overcome this weakness is to create an **agent**, a program powered by an LLM. The agent is empowered by **tools** to help it perform its action.

Thus when during problem-solving the agent needs a specific skill, it can just rely on an appropriate tool from its toolbox.

Experimentally, the agent framework works extremely well, often allowing to obtain state-of-the-art performance on benchmarks. For instance, have a look at [[list benchmarks]].

## Our approach

Building agent workflows is complex, and we feel these systems need a lot of clarity and modularity. We launched Transformers Agents one year ago, and still believe our approach has a lot of value, so we’re doubling down.

Our frameworks strives for:

- **Clarity through simplicity:** we reduce abstractions to the minimum. Simple error logs and accessible attributes let you easily inspect what’s happening, thus leaving you more clarity
- **Modularity:** We prefer to propose building blocks rather than full sets, to let you choose whatever building blocks are best for your project.

On top of that, we have **sharing features** that let you build on the shoulders of giants!

### Main elements

- `Tool`: this is the class that lets you define a tool. It is composed mainly of a forward `method` that executes the tool call, and attributes `name`, `descriptions`, `inputs` and `output_type`  that are used to dynamically generate a usage manual for the tool and insert it into the LLM’s prompt.
- `Toolbox`: We would have preferred to avoid creating that abstraction, but since some tools take time to initialize, it’s better to re-use a previous toolbox and just swap one tool rather than re-building a set of tools from scratch at each agent initialization.
- `CodeAgent`: a very simple agent that generates its actions as one single blob of Python code. It will not be able to iterate on previous observations.
- `ReactAgent`: ReAct agents follow a cycle of Thought ⇒ Action ⇒ Observation until they’ve solve the task. We propose two classes of ReactAgent:
    - `ReactCodeAgent` generates its actions as code blobs.
    - `ReactJsonAgent` generates its actions as JSON blobs.

Check out the doc at [INSERT_LINK] to learn how to use each component!

How do our agents work under the hood?

What’s in common is that in the essence, an agent is “allowing an LLM to use tools”, you will need these key elements in your central `agent.run()` function:

- Provide information about tool usage to your LLM in a **specific prompt**
- **Parse** the tool calls from the LLM output (can be via code, JSON format, or any other format)
- **Execute** calls
- If you want the agent to iterate on previous outputs, you need to **keep a memory** with previous tool calls and observations (can be more or less fine-grained depending on how long-term you want it to be).


<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agents/agent_single_multistep.png" alt="graph of agent workflows" width=90%>
</p>


For more general context about agents, you could read [this excellent blog post](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng or [our earlier blog post](https://huggingface.co/blog/open-source-llms-as-agents) about building agents with LangChain.

## Example use cases

### Self-correcting Retrieval-Augmented-Generation

Quick definition: Retrieval-Augmented-Generation (RAG) is “using an LLM to answer a user query, and letting base its answer on information retrieved from a knowledge base”. It has many advantages over using a vanilla or fine-tuned LLM: to name a few, it allows to ground the answer on true facts and reduce confabulations, it allows to provide the LLM with domain-specific knowledge, and it allows fine-grained control of access to information from the knowledge base.

Let’s say we want to perform RAG, and some parameters must be dynamically generated. For instance, it could be that we need to search only a specific subset of the knowledge base, for instance only the financial documents. Or we want to adapt the number of documents retrieved.

Well, we can do this by giving our agent an access to these parameters!

Let us setup this system. We first load a knowledge base on which we want to perform RAG.

```python
import datasets
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
```

Now we prepare the knowledge base to make it into a vector database for your retriever. We are going to use LangChain, since it features excellent utilities for vector databases:

```python
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

source_docs = [
    Document(
        page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}
    ) for doc in knowledge_base
]

docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(source_docs)[:1000]

embedding_model = HuggingFaceEmbeddings("thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model
)
```

Now that we have the database ready, let’s build an agent that uses it!

We want to leave our agent free to select only from the most relevant sources of information.

Our documents belong to the following sources:

```python
>>> all_sources = list(set([doc.metadata["source"] for doc in docs_processed]))
>>> print(all_sources)

['blog', 'optimum', 'datasets-server', 'datasets', 'transformers', 'course',
'gradio', 'diffusers', 'evaluate', 'deep-rl-class', 'peft',
'hf-endpoints-documentation', 'pytorch-image-models', 'hub-docs']
```

How can we select the relevant sources based on the user query?

👉 Let us leave our agent in charge of this! We create a retriever tool that the agent can call with the parameters of its choice:

```python
import json
from transformers.agents import Tool
from langchain_core.vectorstores import VectorStore

class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "source": {
            "type": "text", 
            "description": ""
        },
    }
    output_type = "text"
    
    def __init__(self, vectordb: VectorStore, all_sources: str, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.inputs["source"]["description"] = (
	        f"The source of the documents to search, as a str representation of a list. Possible values in the list are: {all_sources}. If this argument is not provided, all sources will be searched."
	      )

    def forward(self, query: str, source: str = None) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        if source:
            if isinstance(source, str) and "[" not in str(source): # if the source is not representing a list
                source = [source]
            source = json.loads(str(source).replace("'", '"'))

        docs = self.vectordb.similarity_search(query, filter=({"source": source} if source else None), k=3)

        if len(docs) == 0:
            return "No documents found with this filtering. Try removing the source filter."
        return "Retrieved documents:\n\n" + "\n===Document===\n".join(
            [doc.page_content for doc in docs]
        )
```

Now it’s straightforward to create an agent that leverages this tool!

By the way, the tool could have been loaded from the hub. The documentation shows how to use it, basically: 

```python
from transformers.agents import HfEngine, ReactJsonAgent

llm_engine = HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")

agent = ReactJsonAgent(tools=[RetrieverTool(vectordb, all_sources)], llm_engine=llm_engine)

agent_output = agent.run("Please show me a LORA finetuning script")

print("Final output:")
print(agent_output)
```

```
Calling tool: retriever with arguments: {'query': 'LORA finetuning script', 'source': "['transformers', 'datasets-server', 'datasets']"}
Calling tool: retriever with arguments: {'query': 'LORA finetuning script'}
Calling tool: retriever with arguments: {'query': 'LORA finetuning script example', 'source': "['transformers', 'datasets-server', 'datasets']"}
Calling tool: retriever with arguments: {'query': 'LORA finetuning script example'}
Calling tool: final_answer with arguments: {'answer': 'Here is an example of a LORA finetuning script: https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L371'}

Final output:
Here is an example of a LORA finetuning script: https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py#L371
```

We can see the self-correction in action: the agent first tried to restrict sources, but due to the lack of corresponding documents it ended up not restricting sources at all.

We can verify that by inspecting the llm output at the logs for step 2: `print(agent.logs[2]['llm_output'])`

```
Thought: I'll try to retrieve some documents related to LORA finetuning scripts from the entire knowledge base, without any source filtering.

Action:
{
  "action": "retriever",
  "action_input": {"query": "LORA finetuning script"}
}
```

### Using a simple multi-agent setup 🤝 for efficient web browsing

We want to build an agent and test it on the GAIA benchmark ([Mialon et al. 2023](https://huggingface.co/papers/2311.12983)). GAIA is an extremely difficult benchmark, with most questions requiring several steps of reasoning using different tools. A specifically difficult requirement is to have a powerful web browser, able to navigate to pages with specific constraints: discovering pages using the website’s inner navigation, selecting specific articles in time...

Since this browsing requires diving deeper into subpages, thus scrolling through lots of text tokens that will not necessary for the higher-level task-solving, we prefer to give the web-browsing sub-tasks to a specialized web surfer agent. We provide it with some tools to browse the web and a specific prompt (check the repo to find specific implementations).

```python
from transformers.agents import ReactJsonAgent, HfEngine

WEB_TOOLS = [
    SearchInformationTool(),
    NavigationalSearchTool(),
    VisitTool(),
    DownloadTool(),
    PageUpTool(),
    PageDownTool(),
    FinderTool(),
    FindNextTool(),
]

websurfer_llm_engine = HfEngine(
    model="CohereForAI/c4ai-command-r-plus"
)  # We choose Command-R+ for its high context length

websurfer_agent = ReactJsonAgent(
    tools=WEB_TOOLS,
    llm_engine=websurfer_llm_engine,
)
```

To allow this agent to be called by a higher-level task solving agent, we can simply encapsulate it in another tool:

```python
class SearchTool(Tool):
    name = "ask_search_agent"
    description = "A search agent that will browse the internet to answer a question. Use it to gather informations, not for problem-solving."

    inputs = {
        "question": {
            "description": "Your question, as a natural language sentence. You are talking to an agent, so provide them with as much context as possible.",
            "type": "text",
        }
    }
    output_type = "text"

    def forward(self, question: str) -> str:
        return surfer_agent.run(question)
```

Then we initialize the task-solving agent this search tool and a specific prompt (basically we added a few modifications over the default prompt for a `ReactCodeAgent` , but you still could use the base prompt, it works well):

```python
from transformers.agents import ReactCodeAgent

llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
react_agent_hf = ReactCodeAgent(
    llm_engine=llm_engine,
    tools=[SearchTool()],
)
```

Let's run the agent with the following task:

> *Use density measures from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023.*
> *I have a gallon of honey and a gallon of mayonnaise at 25C. I remove one cup of honey at a time from the gallon of honey. How many times will I need to remove a cup to have the honey weigh less than the mayonaise? Assume the containers themselves weigh the same.*
> 

```
Thought: I will use the 'ask_search_agent' tool to find the density of honey and mayonnaise at 25C.
==== Agent is executing the code below:
density_honey = ask_search_agent(question="What is the density of honey at 25C?")
print("Density of honey:", density_honey)
density_mayo = ask_search_agent(question="What is the density of mayonnaise at 25C?")
print("Density of mayo:", density_mayo)
===
Observation:
Density of honey: The density of honey is around 1.38-1.45kg/L at 20C. Although I couldn't find information specific to 25C, minor temperature differences are unlikely to affect the density that much, so it's likely to remain within this range.
Density of mayo: The density of mayonnaise at 25°C is 0.910 g/cm³.

===== New step =====
Thought: I will convert the density of mayonnaise from g/cm³ to kg/L and then calculate the initial weights of the honey and mayonnaise in a gallon. After that, I will calculate the weight of honey after removing one cup at a time until it weighs less than the mayonnaise.
==== Agent is executing the code below:
density_honey = 1.42 # taking the average of the range
density_mayo = 0.910 # converting g/cm³ to kg/L
density_mayo = density_mayo * 1000 / 1000 # conversion

gallon_to_liters = 3.785 # conversion factor
initial_honey_weight = density_honey * gallon_to_liters
initial_mayo_weight = density_mayo * gallon_to_liters

cup_to_liters = 0.236 # conversion factor
removed_honey_weight = cup_to_liters * density_honey
===
Observation:

===== New step =====
Thought: Now that I have the initial weights of honey and mayonnaise, I'll try to calculate the number of cups to remove from the honey to make it weigh less than the mayonnaise using a simple arithmetic operation.
==== Agent is executing the code below:
cups_removed = int((initial_honey_weight - initial_mayo_weight) / removed_honey_weight) + 1
print("Cups removed:", cups_removed)
final_answer(cups_removed)
===
>>> Final answer: 6
```

✅ And the answer is **correct**!

## Testing our agents

Let’s take our agent framework for a spin and benchmark different models with it!

All the code for the experiments below can be found [here](https://github.com/aymeric-roucher/agent_reasoning_benchmark).

### Comparing LLM engines

The `agents_reasoning_benchmark` is a small - but mighty- reasoning test for evaluating agent performance. The idea is that the choice the tools you use with your agents can radically alter performance for certain tasks. So this benchmark restricts the set of tools used to a calculator and a basic search tool. And we picked questions from several datasets that could be solved using only these two tools

- **30 questions from [HotpotQA](https://huggingface.co/datasets/hotpot_qa)** ([Yang et al., 2018](https://huggingface.co/papers/1809.09600)) ****to test search tool usage.
- **40 questions from [GSM8K](https://huggingface.co/datasets/gsm8k)** ([Cobbe et al., 2021](https://huggingface.co/papers/2110.14168))to test calculator usage.
- **20 questions from [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)** ([Mialon et al., 2023](https://huggingface.co/papers/2311.12983)) to test the usage of both tools for solving difficult questions.

This benchmark was already used (in an alpha version) and explained in more detail in [our earlier blog post](https://huggingface.co/blog/open-source-llms-as-agents). Here we try 3 different engines: [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), and [GPT-4 Turbo](https://platform.openai.com/docs/models).

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agents/aggregate_score.png" alt="benchmark of agent performances" width=90%>
</p>

The results are shown above - as the average of two complete runs for more precision. We also tested [Command-R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) and [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1), but do not show them for clarity.

**Llama3-70B-Instruct leads the Open-Source models: it is on par with GPT-4**, and it’s especially strong in a `ReactCodeAgent`  thanks to Llama3’s strong coding performance!

Comparing JSON- and Code-based agents is interesting: for less powerful models like Mixtral-8x7B, Code-based agents do not perform well, since the LLM often fails to generate good code. But this version catches up at higher levels: in our experience, on the full GAIA test with Llama3-70B-Instruct, the Code version even outperforms the JSON. So that is what we used for the next step: testing on the complete GAIA benchmark.

### Climbing up the GAIA Leaderboard with a multi-modal agent

[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) ([Mialon et al., 2023](https://huggingface.co/papers/2311.12983)) is an extremely difficult benchmark. You can see in the `agent_reasoning_benchmark` above that models still do not perform well although we cherry-picked tasks that they could solve with basic tools. To get food performance we need an agent with specific tools:

- `SearchTool`: the web browser defined above.
- `TextInspectorTool`: open documents as text files and return their content.
- `SpeechToTextTool`: transcribe audio files to text. We use the default tool based on [distil-whisper](https://huggingface.co/distil-whisper/distil-large-v3).
- `VisualQATool`: visualize images. For these we use the shiny new [Idefics2-8b-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty)!

Inspect the code in the [repo](https://github.com/aymeric-roucher/agent_reasoning_benchmark) for more detail on how we define these tools. Then we initialize our agent:

```python
from transformers.agents import ReactCodeAgent, HfEngine

TASK_SOLVING_TOOLBOX = [
    SearchTool(),
    VisualQATool(),
    SpeechToTextTool(),
    TextInspectorTool(),
]

react_agent_hf = ReactCodeAgent(
    llm_engine=HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct"),
    tools=TASK_SOLVING_TOOLBOX,
    memory_verbose=True,
)
```

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agents/leaderboard.png" alt="GAIA leaderboard" width=90%>
</p>

And after some time needed to complete the 165 questions, we submit our result to the [GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard), and… 🥁🥁🥁 
⇒ Our agent comes at the 4th place: it beats many GPT4-based agents, and is now the reigning contender for the Open-Source category!


## Conclusion

We will keep improving this package in the coming months. We have already identified several exciting paths:

- agent sharing options
- better tools (especially in image processing)
- long-term memory management
- multi-agent collaboration

👉 **Go try out transformers agents!** We’re looking forward to receiving your feedback and your ideas. Let’s fill the top of the leaderboard with more open-source models! 🚀