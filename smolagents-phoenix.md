---
title: "Trace & Evaluate your Agent with Arize Phoenix"
thumbnail: /blog/assets/smolagents-phoenix/thumbnail.jpg
authors:
- user: schavalii
  guest: true
  org: arize-ai
- user: jgilhuly16
  guest: true
  org: arize-ai
- user: m-ric
---

# Trace & Evaluate your Agent with Arize Phoenix

So, you’ve built your agent. It takes in inputs and tools, processes them, and generates responses. Maybe it’s making decisions, retrieving information, executing tasks autonomously, or all three. But now comes the big question – how effectively is it performing? And more importantly, how do you know? 

Building an agent is one thing; understanding its behavior is another. That’s where tracing and evaluations come in. Tracing allows you to see exactly what your agent is doing step by step—what inputs it receives, how it processes information, and how it arrives at its final output. Think of it like having an X-ray for your agent’s decision-making process. Meanwhile, evaluation helps you measure performance, ensuring your agent isn’t just functional, but actually effective. Is it producing the right answers? How relevant are its findings at  each step? How well-crafted is the agent’s response? Does it align with your goals?

[Arize Phoenix](https://phoenix.arize.com/) provides a centralized platform to trace, evaluate, and debug your agent's decisions in real time—all in one place. We’ll dive into how you can implement them to refine and optimize your agent. Because building is just the beginning—true intelligence comes from knowing exactly what’s happening under the hood.

For this, let’s make sure that we have an Agent setup\! You can follow along with the following steps or use your own agent. 

## Make An Agent

### Step 1: Install the Required Libraries 

```py
pip install -q smolagents
```

### Step 2: Import all the Essential Building Blocks 

Now let’s bring in the classes and tools we’ll be using:

```py
from smolagents import (
   CodeAgent,
   DuckDuckGoSearchTool,
   VisitWebpageTool,
   HfApiModel,
)
```

### Step 3: Set Up Our Base Models 

We’ll create a model instance powered by the Hugging Face Hub Serverless API:

```py
hf_model = HfApiModel()
```

### Step 4: Create the Tool-Calling Agent

```py
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=hf_model,
    add_base_tools=True
)
```

### Step 5: Run the Agent

Now for the magic moment—let’s see our agent in action. The question we’re asking our agent is:  
*“Fetch the share price of Google from 2020 to 2024, and create a line graph from it?”*

```py
agent.run("fetch the share price of google from 2020 to 2024, and create a line graph from it?")
```

Your agent will now:

1. Use DuckDuckGoSearchTool to search for historical share prices of Google.  
2. Potentially visit pages with the VisitWebpageTool to find that data.  
3. Attempt to gather information and generate or describe how to create the line graph.

## Trace Your Agent

Once your agent is running, the next challenge is making sense of its internal workflow. Tracing helps you track each step your agent takes—from invoking tools to processing inputs and generating responses—allowing you to debug issues, optimize performance, and ensure it behaves as expected.

To enable tracing, we’ll use Arize Phoenix for visualization, and OpenTelemetry \+ OpenInference for instrumentation.

Install the telemetry module from smolagents:

```py
pip install -q 'smolagents[telemetry]'
```

You can run Phoenix in a bunch of different ways. This command will run a local instance of Phoenix on your machine:

```py
python -m phoenix.server.main serve
```

For other hosting options of Phoenix, you can [create a free online instance of Phoenix](https://app.phoenix.arize.com/login/sign-up), [self-host the application locally](https://docs.arize.com/phoenix/deployment), or [host the application on Hugging Face Spaces](https://huggingface.co/learn/cookbook/en/phoenix_observability_on_hf_spaces).

After launching, we register a tracer provider, pointing to our Phoenix instance.

```py
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

tracer_provider = register(project_name="my-smolagents-app") # creates a tracer provider to capture OTEL traces
SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider) # automatically captures any smolagents calls as traces
```

Now any calls made to smolagents will be sent through to our Phoenix instance.


Now that tracing is enabled, let’s test it with a simple query:

```py
agent.run("What time is it in Tokyo right now?")
```

Once OpenInference is set up with SmolAgents, every agent invocation will be automatically traced in Phoenix.  


<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/smolagents-phoenix/smolagentsBlogTracing.gif" />
</div>


## Evaluate Your Agent

Once your agent is up and its run is monitored, the next step is to assess its performance. Evaluations (evals) help determine how well your agent is retrieving, processing, and presenting information.

There are many types of evals you can run, such as response relevance, factual accuracy, latency, and more. Check out the [Phoenix documentation](https://docs.arize.com/phoenix/evaluation/llm-evals) for a deeper dive into different evaluation techniques.

In this example, we’ll focus on evaluating the DuckDuckGo search tool used by our agent. We’ll measure the relevance of its search results using a Large Language Model (LLM) as a judge—specifically, OpenAI's GPT-4o.

### Step 1: Install OpenAI  

First, install the necessary packages:

```py
pip install -q openai
```

We’ll be using GPT-4o to evaluate whether the search tool’s responses are relevant.   
This method, known as LLM-as-a-judge, leverages language models to classify and score responses.

### Step 2: Retrieve Tool Execution Spans 

To evaluate how well DuckDuckGo is retrieving information, we first need to extract the execution traces where the tool was called.

```py
from phoenix.trace.dsl import SpanQuery
import phoenix as px
import json

query = SpanQuery().where(
    "name == 'DuckDuckGoSearchTool'",
).select(
    input="input.value", # this parameter must be named input to work with the RAG_RELEVANCY_PROMPT_TEMPLATE
    reference="output.value", # this parameter must be named reference to work with the RAG_RELEVANCY_PROMPT_TEMPLATE
)

# The Phoenix Client can take this query and return the dataframe.
tool_spans = px.Client().query_spans(query, project_name="my-smolagents-app")

tool_spans["input"] = tool_spans["input"].apply(lambda x: json.loads(x).get("kwargs", {}).get("query", ""))
tool_spans.head()
```

### Step 3: Import Prompt Template  

Next, we load the RAG Relevancy Prompt Template, which will help the LLM classify whether the search results are relevant or not.

```py
from phoenix.evals import (
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    OpenAIModel,
    llm_classify
)
import nest_asyncio
nest_asyncio.apply()

print(RAG_RELEVANCY_PROMPT_TEMPLATE)
```

### Step 4: Run the Evaluation  

Now, we run the evaluation using **GPT-4o** as the judge:

```py
from phoenix.evals import (
    llm_classify,
    OpenAIModel,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
)

eval_model = OpenAIModel(model="gpt-4o")

eval_results = llm_classify(
    dataframe=tool_spans,
    model=eval_model,
    template=RAG_RELEVANCY_PROMPT_TEMPLATE,
    rails=["relevant", "unrelated"],
    concurrency=10,
    provide_explanation=True,
)
eval_results["score"] = eval_results["explanation"].apply(lambda x: 1 if "relevant" in x else 0)
```

What’s happening here?

* We use GPT-4o to analyze the search query (input) and search result (output).  
* The LLM classifies whether the result is relevant or unrelated based on the prompt.  
* We assign a binary score (1 \= relevant, 0 \= unrelated) for further analysis.

To see your results: 

```py
eval_results.head()
```

### Step 5: Send Evaluation Results to Phoenix

```py
from phoenix.trace import SpanEvaluations

px.Client().log_evaluations(SpanEvaluations(eval_name="DuckDuckGoSearchTool Relevancy", dataframe=eval_results))
```

With this setup, we can now systematically evaluate the effectiveness of the DuckDuckGo search tool within our agent. Using LLM-as-a-judge, we can ensure our agent retrieves accurate and relevant information, leading to better performance.  
Any evaluation is easy to set up using this tutorial—just swap out the RAG\_RELEVANCY\_PROMPT\_TEMPLATE for a different prompt template that fits your needs. Phoenix provides a variety of pre-written and pre-tested evaluation templates, covering areas like faithfulness, response coherence, factual accuracy, and more. Check out the Phoenix docs to explore the full list and find the best fit for your agent\! 

| Evaluation Template | Applicable Agent Type  |
| :---- | :---- |
| Hallucination Detection | RAG agents  General chatbots Knowledge-based assistants |
| Q\&A on Retrieved Data | RAG agents Research Assistants Document Search Tools |
| RAG Relevance | RAG agents Search-based AI assistants |
| Summarization | Summarization tools Document digesters Meeting note generators |
| Code Generation | Code assistants AI programming bots |
| Toxicity Detection | Moderation bots Content filtering AI |
| AI vs Human (Ground Truth) | Evaluation & benchmarking tools AI-generated content validators |
| Reference (Citation) Link | Research assistants Citation tools Academic writing aids |
| SQL Generation Evaluation | Database query agents SQL automation tools |
| Agent Function Calling Evaluation | Multi-step reasoning agents API-calling AI Task automation bots |

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/smolagents-phoenix/smolagentEvals.gif" />
</div>
