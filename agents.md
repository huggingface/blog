---
title: "License to Call: Introducing Transformers Agents 2.0"
thumbnail: /blog/assets/agents/thumbnail.png
authors:
  - user: m-ric
  - user: lysandre
  - user: pcuenq
---

# License to Call: Introducing Transformers Agents 2.0

## TL;DR

We are releasing Transformers Agents 2.0!

⇒ 🎁 On top of our existing agent type, we introduce two new agents that **can iterate based on past observations to solve complex tasks**.

⇒ 💡 We aim for the code to be **clear and modular, and for common attributes like the final prompt and tools to be transparent**.

⇒ 🤝 We add **sharing options** to boost community agents.

⇒ 💪 **Extremely performant new agent framework**, allowing a Llama-3-70B-Instruct agent to outperform GPT-4 based agents in the GAIA Leaderboard!

🚀 Go try it out and climb ever higher on the GAIA leaderboard!

## Table of Contents

- [What is an agent?](#what-is-an-agent)
- [The Transformers Agents approach](#the-transformers-agents-approach)
    - [Main elements](#main-elements)
- [Example use-cases](#example-use-cases)
    - [Self-correcting Retrieval-Augmented-Generation](#self-correcting-retrieval-augmented-generation)
    - [Using a simple multi-agent setup 🤝 for efficient web browsing](#using-a-simple-multi-agent-setup-for-efficient-web-browsing)
- [Testing our agents](#testing-our-agents)
    - [Benchmarking LLM engines](#benchmarking-llm-engines)
    - [Climbing up the GAIA Leaderboard with a multi-modal agent](#climbing-up-the-gaia-leaderboard-with-a-multi-modal-agent)
- [Conclusion](#conclusion)

## What is an agent?

Large Language Models (LLMs) can tackle a wide range of tasks, but they often struggle with specific tasks like logic, calculation, and search. When prompted in these domains in which they do not perform well, they frequently fail to generate a correct answer.

One approach to overcome this weakness is to create an **agent**, which is just a program driven by an LLM. The agent is empowered by **tools** to help it perform actions. When the agent needs a specific skill to solve a particular problem, it relies on an appropriate tool from its toolbox.

Thus when during problem-solving the agent needs a specific skill, it can just rely on an appropriate tool from its toolbox.

Experimentally, agent frameworks generally work very well, achieving state-of-the-art performance on several benchmarks. For instance, have a look at [the top submissions for HumanEval](https://paperswithcode.com/sota/code-generation-on-humaneval): they are agent systems.

## The Transformers Agents approach

Building agent workflows is complex, and we feel these systems need a lot of clarity and modularity. We launched Transformers Agents one year ago, and we’re doubling down on our core design goals.

Our framework strives for:

- **Clarity through simplicity:** we reduce abstractions to the minimum. Simple error logs and accessible attributes let you easily inspect what’s happening and give you more clarity.
- **Modularity:** We prefer to propose building blocks rather than full, complex feature sets. You are free to choose whatever building blocks are best for your project.
    - For instance, since any agent system is just a vehicle powered by an LLM engine, we decided to conceptually separate the two, which lets you create any agent type from any underlying LLM.

On top of that, we have **sharing features** that let you build on the shoulders of giants!

### Main elements

- `Tool`: this is the class that lets you use a tool or implement a new one. It is composed mainly of a callable forward `method` that executes the tool action, and a set of a few essential attributes: `name`, `descriptions`, `inputs` and `output_type`. These attributes are used to dynamically generate a usage manual for the tool and insert it into the LLM’s prompt.
- `Toolbox`: It's a set of tools that are provided to an agent as resources to solve a particular task. For performance reasons, tools in a toolbox are already instantiated and ready to go. This is because some tools take time to initialize, so it’s usually better to re-use an existing toolbox and just swap one tool, rather than re-building a set of tools from scratch at each agent initialization.
- `CodeAgent`: a very simple agent that generates its actions as one single blob of Python code. It will not be able to iterate on previous observations.
- `ReactAgent`: ReAct agents follow a cycle of Thought ⇒ Action ⇒ Observation until they’ve solve the task. We propose two classes of ReactAgent:
    - `ReactCodeAgent` generates its actions as python blobs.
    - `ReactJsonAgent` generates its actions as JSON blobs.

Check out [the documentation](https://huggingface.co/docs/transformers/en/main_classes/agent) to learn how to use each component!

How do agents work under the hood?

In essence, what an agent does is “allowing an LLM to use tools”. Agents have a key `agent.run()` method that:

- Provides information about tool usage to your LLM in a **specific prompt**. This way, the LLM can select tools to run to solve the task.
- **Parses** the tool calls from the LLM output (can be via code, JSON format, or any other format).
- **Executes** the calls.
- If the agent is designed to iterate on previous outputs, it **keeps a memory** with previous tool calls and observations. This memory can be more or less fine-grained depending on how long-term you want it to be.

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agents/agent_single_multistep.png" alt="graph of agent workflows" width=90%>
</p>


For more general context about agents, you could read [this excellent blog post](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng or [our earlier blog post](https://huggingface.co/blog/open-source-llms-as-agents) about building agents with LangChain.


To take a deeper dive in our package, go take a look at the [agents documentation](https://huggingface.co/docs/transformers/en/transformers_agents).


## Example use cases

In order to get access to the early access of this feature, please first install `transformers` from its `main` branch:
```
pip install "git+https://github.com/huggingface/transformers.git#egg=transformers[agents]"
```
Agents 2.0 will be released in the v4.41.0 version, landing mid-May.


### Self-correcting Retrieval-Augmented-Generation

Quick definition: Retrieval-Augmented-Generation (RAG) is “using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base”. It has many advantages over using a vanilla or fine-tuned LLM: to name a few, it allows to ground the answer on true facts and reduce confabulations, it allows to provide the LLM with domain-specific knowledge, and it allows fine-grained control of access to information from the knowledge base.

Let’s say we want to perform RAG, and some parameters must be dynamically generated. For example, depending on the user query we could want to restrict the search to specific subsets of the knowledge base, or we could want to adjust the number of documents retrieved. The difficulty is: how to dynamically adjust these parameters based on the user query?

Well, we can do this by giving our agent an access to these parameters!

Let's setup this system. 

Tun the line below to install required dependancies:
```
pip install langchain sentence-transformers faiss-cpu
```

We first load a knowledge base on which we want to perform RAG: this dataset is a compilation of the documentation pages for many `huggingface` packages, stored as markdown.

```python
import datasets
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
```

Now we prepare the knowledge base by processing the dataset and storing it into a vector database to be used by the retriever. We are going to use LangChain, since it features excellent utilities for vector databases:

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

Now that we have the database ready, let’s build a RAG system that answers user queries based on it!

We want our system to select only from the most relevant sources of information, depending on the query.

Our documentation pages come from the following sources:

```python
>>> all_sources = list(set([doc.metadata["source"] for doc in docs_processed]))
>>> print(all_sources)

['blog', 'optimum', 'datasets-server', 'datasets', 'transformers', 'course',
'gradio', 'diffusers', 'evaluate', 'deep-rl-class', 'peft',
'hf-endpoints-documentation', 'pytorch-image-models', 'hub-docs']
```

How can we select the relevant sources based on the user query?

👉 Let us build our RAG system as an agent that will be free to choose its sources!

We create a retriever tool that the agent can call with the parameters of its choice:

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

The agent will need these arguments upon initialization:
- *`tools`*: a list of tools that the agent will be able to call.
- *`llm_engine`*: the LLM that powers the agent.

Our `llm_engine` must be a callable that takes as input a list of [messages](https://huggingface.co/docs/transformers/main/chat_templating) and returns text. It also needs to accept a `stop_sequences` argument that indicates when to stop its generation. For convenience, we directly use the `HfEngine` class provided in the package to get a LLM engine that calls our [Inference API](https://huggingface.co/docs/api-inference/en/index).

```python
from transformers.agents import HfEngine, ReactJsonAgent

llm_engine = HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")

agent = ReactJsonAgent(
    tools=[RetrieverTool(vectordb, all_sources)],
    llm_engine=llm_engine
)

agent_output = agent.run("Please show me a LORA finetuning script")

print("Final output:")
print(agent_output)
```

Since we initialized the agent as a `ReactJsonAgent`, it has been automatically given a default system prompt that tells the LLM engine to process step-by-step and generate tool calls as JSON blobs (you could replace this prompt template with your own as needed).

Then when its `.run()` method is launched, the agent takes care of calling the LLM engine, parsing the tool call JSON blobs and executing these tool calls, all in a loop that ends only when the final answer is provided.

And we get the following output:

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

In this example, we want to build an agent and test it on the GAIA benchmark ([Mialon et al. 2023](https://huggingface.co/papers/2311.12983)). GAIA is an extremely difficult benchmark, with most questions requiring several steps of reasoning using different tools. A specifically difficult requirement is to have a powerful web browser, able to navigate to pages with specific constraints: discovering pages using the website’s inner navigation, selecting specific articles in time...

Web browsing requires diving deeper into subpages and scrolling through lots of text tokens that will not be necessary for the higher-level task-solving. We assign the web-browsing sub-tasks to a specialized web surfer agent. We provide it with some tools to browse the web and a specific prompt (check the repo to find specific implementations).

Defining these tools is outside the scope of this post: but you can check [the repository](https://github.com/aymeric-roucher/agent_reasoning_benchmark) to find specific implementations.

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
        return websurfer_agent.run(question)
```

Then we initialize the task-solving agent with this search tool:

```python
from transformers.agents import ReactCodeAgent

llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
react_agent_hf = ReactCodeAgent(
    tools=[SearchTool()],
    llm_engine=llm_engine,
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

### Benchmarking LLM engines

The `agents_reasoning_benchmark` is a small - but mighty- reasoning test for evaluating agent performance. This benchmark was already used and explained in more detail in [our earlier blog post](https://huggingface.co/blog/open-source-llms-as-agents). 

The idea is that the choice of tools you use with your agents can radically alter performance for certain tasks. So this benchmark restricts the set of tools used to a calculator and a basic search tool. We picked questions from several datasets that could be solved using only these two tools:

- **30 questions from [HotpotQA](https://huggingface.co/datasets/hotpot_qa)** ([Yang et al., 2018](https://huggingface.co/papers/1809.09600)) to test search tool usage.
- **40 questions from [GSM8K](https://huggingface.co/datasets/gsm8k)** ([Cobbe et al., 2021](https://huggingface.co/papers/2110.14168)) to test calculator usage.
- **20 questions from [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)** ([Mialon et al., 2023](https://huggingface.co/papers/2311.12983)) to test the usage of both tools for solving difficult questions.

Here we try 3 different engines: [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), and [GPT-4 Turbo](https://platform.openai.com/docs/models).

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agents/aggregate_score.png" alt="benchmark of agent performances" width=90%>
</p>

The results are shown above - as the average of two complete runs for more precision. We also tested [Command-R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) and [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1), but do not show them for clarity.

⇒ **Llama-3-70B-Instruct leads the Open-Source models: it is on par with GPT-4**, and it’s especially strong in a `ReactCodeAgent`  thanks to Llama 3’s strong coding performance!

💡 It's interesting to compare JSON- and Code-based React agents: with less powerful LLM engines like Mixtral-8x7B, Code-based agents do not perform as well as JSON, since the LLM engine frequently fails to generate good code. But the Code version really shines with more powerful models as engines: in our experience, the Code version even outperforms the JSON with Llama-3-70B-Instruct. As a result, we use the Code version for our next challenge: testing on the complete GAIA benchmark.

### Climbing up the GAIA Leaderboard with a multi-modal agent

[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) ([Mialon et al., 2023](https://huggingface.co/papers/2311.12983)) is an extremely difficult benchmark: you can see in the `agent_reasoning_benchmark` above that models do not perform above 50% even though we cherry-picked tasks that could be solved with 2 basic tools.

Now we want to get a score on the complete set, we do not cherry-pick questions anymore. Thus we have to cover all modalities, which leads us to use these specific tools:
- `SearchTool`: the web browser defined above.
- `TextInspectorTool`: open documents as text files and return their content.
- `SpeechToTextTool`: transcribe audio files to text. We use the default tool based on [distil-whisper](https://huggingface.co/distil-whisper/distil-large-v3).
- `VisualQATool`: analyze images visually. For these we use the shiny new [Idefics2-8b-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty)!

We first initialize these toole (for more detail, inspect the code in the [repository](https://github.com/aymeric-roucher/agent_reasoning_benchmark)).

Then we initialize our agent:

```python
from transformers.agents import ReactCodeAgent, HfEngine

TASK_SOLVING_TOOLBOX = [
    SearchTool(),
    VisualQATool(),
    SpeechToTextTool(),
    TextInspectorTool(),
]

react_agent_hf = ReactCodeAgent(
    tools=TASK_SOLVING_TOOLBOX,
    llm_engine=HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct"),
    memory_verbose=True,
)
```

And after some time needed to complete the 165 questions, we submit our result to the [GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard), and… 🥁🥁🥁

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agents/leaderboard.png" alt="GAIA leaderboard" width=90%>
</p>

⇒ Our agent ranks 4th: it beats many GPT-4-based agents, and is now the reigning contender for the Open-Source category!


## Conclusion

We will keep improving this package in the coming months. We have already identified several exciting paths in our development roadmap:
- More agent sharing options: for now you can push or load tools from the Hub, we will implement pushing/loading agents too.
- Better tools, especially for image processing.
- Long-term memory management.
- Multi-agent collaboration.

👉 **Go try out transformers agents!** We’re looking forward to receiving your feedback and your ideas.

Let’s fill the top of the leaderboard with more open-source models! 🚀