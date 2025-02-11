---
title: Agent Leaderboard
thumbnail: /blog/assets/agent-leaderboard/blog_cover_tiny.png
authors:
- user: pratikbhavsar
---

# Agent Leaderboard: Evaluating AI Agent Performance in Multi-Domain Scenarios

<div align="center">

[![Leaderboard](https://img.shields.io/badge/🤗%20View-Leaderboard-red)](https://huggingface.co/spaces/galileo-ai/agent-leaderboard)
[![GitHub](https://img.shields.io/badge/📂%20GitHub-Repository-blue)](https://github.com/rungalileo/agent-leaderboard)
[![Dataset](https://img.shields.io/badge/🔍%20Explore-Dataset-purple)](https://huggingface.co/datasets/galileo-ai/agent-leaderboard)

</div>

Jensen Huang has called AI agents the "digital workforce" - and he is not the only tech CEO who thinks agents represent the next significant breakthrough for AI. Satya Nadella believes agents will fundamentally transform how businesses operate.

These agents can interact with external tools and APIs, dramatically expanding their practical applications. However, they are far from perfect, and evaluating their performance in this domain has been challenging due to the complexity of potential interactions.

Our Agent Leaderboard evaluates agent performance using Galileo's tool selection quality metric to clearly understand how different LLMs handle tool-based interactions across various dimensions.

We built this leaderboard to answer a straightforward question: "How do AI agents perform in real-world business scenarios?" While academic benchmarks tell us about technical capabilities, we want to know which models work for varied use cases.

<script
 type="module"
 src="https://gradio.s3-us-west-2.amazonaws.com/5.5.0/gradio.js"
></script>

<gradio-app theme_mode="light" space="galileo-ai/agent-leaderboard"></gradio-app>

## What sets this Agent Evaluation Leaderboard apart?

<p align="center">
  <img src="https://github.com/rungalileo/agent-leaderboard/raw/main/images/overview.png" />
</p>

Current evaluation frameworks address specific niches. BFCL excels in academic domains like mathematics, entertainment, and education, τ-bench specializes in retail and airline scenarios, xLAM covers data generation across 21 domains, and ToolACE focuses on API interactions in 390 domains. Our leaderboard synthesizes these datasets into a comprehensive evaluation framework that spans multiple domains and real-world use cases.

We provide actionable insights into how these models handle edge cases and safety considerations by incorporating various benchmarks and testing scenarios. We analyze cost-effectiveness, provide implementation guidance, and assess business impact – factors crucial for organizations deploying AI agents. Our leaderboard is designed to help teams decide which models best suit their specific AI agent's needs and constraints.

New LLMs come out very often. We plan to update our benchmark every month to keep this in sync with ongoing model releases.

## Key Insights

Our analysis of 17 leading LLMs revealed intriguing patterns in how AI agents handle real-world tasks. We stress-tested both private and open-source models against 14 diverse benchmarks, measuring everything from simple API calls to complex multi-tool interactions.

<p align="center">
  <img src="https://github.com/rungalileo/agent-leaderboard/raw/main/images/ranking.png" />
</p>

Our findings challenge conventional wisdom about model performance and provide practical insights for teams building with AI agents.

<p align="center">
  <img src="https://github.com/rungalileo/agent-leaderboard/raw/main/images/insights.png" />
</p>

## Complexity of Tool Calling

The complexity of tool calling extends far beyond simple API invocations. Various scenarios challenge AI agents' ability to make appropriate decisions about tool usage:

<p align="center">
  <img src="https://github.com/rungalileo/agent-leaderboard/raw/main/images/complexity_of_tool_calling.png" />
</p>

### Scenario Recognition

When an agent encounters a query, it must first determine if tool usage is warranted. Information may already exist in the conversation history, making tool calls redundant. Alternatively, available tools might be insufficient or irrelevant to the task, requiring the agent to acknowledge limitations rather than force inappropriate tool usage.

### Tool Selection Dynamics

Tool selection isn't binary—it involves both precision and recall. An agent might correctly identify one necessary tool while missing others (recall issue) or select appropriate tools alongside unnecessary ones (precision issue). While suboptimal, these scenarios represent different severity levels of selection errors.

### Parameter Handling

Even with correct tool selection, argument handling introduces additional complexity. Agents must:

- Provide all required parameters with the correct naming
- Handle optional parameters appropriately
- Maintain parameter value accuracy
- Format arguments according to tool specifications

### Sequential Decision Making

Multi-step tasks require agents to:

- Determine optimal tool calling sequence
- Handle interdependencies between tool calls
- Maintain context across multiple operations
- Adapt to partial results or failures

These complexities highlight why tool selection quality shouldn't be viewed as a simple metric but rather as a multifaceted evaluation of an agent's decision-making capabilities in real-world scenarios.

## Methodology

Our evaluation process follows a systematic approach to ensure a comprehensive and fair assessment of AI agents:

### Model Selection

We begin by curating a diverse set of leading language models, including both proprietary and open-source implementations. This selection ensures we capture the full spectrum of available technologies.

### Agent Configuration

Each model is configured as an agent using a standardized system prompt and given access to a consistent set of tools. This standardization ensures that performance differences reflect inherent model capabilities rather than prompt engineering.

### Metric Definition

We established Tool Selection Quality (TSQ) as our primary evaluation metric, focusing on both the correctness of tool selection and the quality of parameter usage. This metric was carefully designed to capture real-world performance requirements.

### Dataset Curation

We strategically sampled from established benchmarking datasets to create a balanced, multi-domain evaluation suite. This dataset tests everything from basic function calls to complex multi-turn interactions, ensuring comprehensive coverage of agent capabilities.

### Scoring System

The final performance score is calculated as an equally weighted average across all datasets. This approach ensures that no single capability dominates the overall assessment, providing a balanced view of agent performance.

Through this structured approach we provide insights that directly translate to real-world implementation decisions.

## How Do We Measure Agent's Performance?

<p align="center">
  <img src="https://github.com/rungalileo/agent-leaderboard/raw/main/images/evaluating_agents.png" />
</p>

### How it works?

As we saw above, tool calling evaluation requires robust measurement across diverse scenarios. We developed the Tool Selection Quality metric to assess agents' tool call performance, evaluating tool selection accuracy and effectiveness of parameter usage. This framework determines whether agents appropriately utilize tools for tasks while also identifying situations where tool usage is unnecessary.

The evaluation uses GPT-4o with ChainPoll to assess tool selection decisions. Multiple independent judgments are gathered for each interaction, with the final score representing the proportion of positive assessments. Each judgment includes a detailed explanation, providing transparency into the evaluation process.

```python
import promptquality as pq

df = pd.read_parquet(file_path, engine="fastparquet")

chainpoll_tool_selection_scorer = pq.CustomizedChainPollScorer(
                scorer_name=pq.CustomizedScorerName.tool_selection_quality,
                model_alias=pq.Models.gpt_4o,
            )

evaluate_handler = pq.GalileoPromptCallback(
        project_name=project_name,
        run_name=run_name,
        scorers=[chainpoll_tool_selection_scorer],
    )

llm = llm_handler.get_llm(model, temperature=0.0, max_tokens=4000)
system_msg = {
            "role": "system",
            "content": 'Your job is to use the given tools to answer the query of human. If there is no relevant tool then reply with "I cannot answer the question with given tools". If tool is available but sufficient information is not available, then ask human to get the same. You can call as many tools as you want. Use multiple tools if needed. If the tools need to be called in a sequence then just call the first tool.',
        }

for row in df.itertuples():
    chain = llm.bind_tools(tools)
    outputs.append(
            chain.invoke(
                [system_msg, *row.conversation], 
                config=dict(callbacks=[evaluate_handler])
            )
        )

evaluate_handler.finish()
```

### Why we need an LLM for tool call evaluation?

The LLM-based evaluation approach enables comprehensive assessment across diverse scenarios. It verifies appropriate handling of insufficient context, recognizing when more information is needed before tool use. For multi-tool scenarios, it checks if all necessary tools are identified and used in the correct sequence. In long-context situations, it ensures relevant information from earlier in the conversation is considered. When tools are absent or inappropriate, it confirms the agent correctly abstains from tool use rather than forcing an incorrect action.

Success in this metric requires sophisticated capabilities: selecting the right tools when needed, providing correct parameters, coordinating multiple tools effectively, and recognizing situations where tool use is unnecessary. For instance, if all required information exists in conversation history or if no suitable tools are available, the correct action is to abstain from tool use.

## What is in the Datasets?

The evaluation framework employs a carefully curated set of benchmarking datasets from BFCL (Berkeley Function Calling Leaderboard), τ-bench (Tau benchmark), Xlam, and ToolACE. Each is designed to test specific aspects of agent capabilities. Understanding these dimensions is necessary for both model evaluation and practical application development.

<p align="center">
  <img src="https://github.com/rungalileo/agent-leaderboard/raw/main/images/datasets.png" />
</p>

### Single-Turn Capabilities

**Basic Tool Usage** scenarios evaluate the fundamental ability to understand tool documentation, handle parameters, and execute basic function calls. This testing dimension focuses on response formatting and error handling in straightforward interactions. This capability is critical for simple automation tasks like setting reminders or fetching basic information in practical applications. [xlam_single_tool_single_call]  

**Tool Selection** scenarios assess a model's ability to choose the right tool from multiple options. This dimension examines how well models understand tool documentation and make decisions about tool applicability. For practical applications, this capability is essential when building multi-purpose agents. [xlam_multiple_tool_single_call]  

**Parallel Execution** scenarios examine a model's capability to orchestrate multiple tools simultaneously. This dimension is particularly critical for efficiency in real-world applications. [xlam_multiple_tool_multiple_call]  

**Tool Reuse** scenarios evaluate the efficient handling of batch operations and parameter variations. This aspect is particularly important for batch processing scenarios in practical applications. [xlam_single_tool_multiple_call]  

### Error Handling and Edge Cases

**Irrelevance Detection** scenarios test a model's ability to recognize tool limitations and communicate appropriately when available tools don't match user needs. This capability is fundamental to user experience and system reliability. [BFCL_v3_irrelevance]  

**Missing Tool Handling** scenarios examine how gracefully models handle situations where required tools are unavailable, including their ability to communicate limitations and suggest alternatives. [xlam_tool_miss, BFCL_v3_multi_turn_miss_func]  

### Context Management

**Long Context** scenarios evaluate a model's ability to maintain context in extended interactions and understand complex instructions. This capability is crucial for complex workflows and extended interactions. [tau_long_context, BFCL_v3_multi_turn_long_context]  

### Multi-Turn Interactions

**Basic Conversation** scenarios test conversational function calling abilities and context retention across turns. This fundamental capability is essential for interactive applications. [BFCL_v3_multi_turn_base_single_func_call, toolace_single_func_call]

**Complex Interaction** scenarios combine multiple challenges to test overall robustness and complex scenario handling. [BFCL_v3_multi_turn_base_multi_func_call, BFCL_v3_multi_turn_composite]  

### Parameter Management

**Missing Parameters** scenarios examine how models handle incomplete information and interact with users to gather necessary parameters. [BFCL_v3_multi_turn_miss_param]  

## Practical Implications for AI Engineers

Our evaluation reveals several key considerations for creating robust and efficient systems when developing AI agents. Let's break down the essential aspects:

<p align="center">
  <img src="https://github.com/rungalileo/agent-leaderboard/raw/main/images/implications.png" />
</p>

### Model Selection and Performance

Advanced models that score above 0.85 in composite tasks are crucial for handling complex workflows, though most models can manage basic tools effectively. When dealing with parallel operations, it's important to examine execution scores for each specific task rather than relying on overall performance metrics.

### Context and Error Management

Implementing context summarization strategies is essential for models that struggle with long-context scenarios. Strong error handling mechanisms become particularly crucial when working with models that show weaknesses in irrelevance detection or parameter handling. Consider implementing structured workflows to guide parameter collection for models that need additional support in this area.

### Safety and Reliability

Implement robust tool access controls, especially for models that struggle to detect irrelevant operations. For models with inconsistent performance, adding validation layers can help maintain reliability. It's also essential to build error recovery systems, particularly for models that have difficulty handling missing parameters.

### Optimizing System Performance

Design your workflow architecture based on each model's ability to handle parallel execution and long-context scenarios. When implementing batching strategies, consider how well the model reuses tools, as this can significantly impact efficiency.

### Current State of AI Models

While proprietary models currently lead in overall capabilities, open-source alternatives are rapidly improving. Simple tool interactions are becoming increasingly reliable across all models, though challenges remain in complex multi-turn interactions and long-context scenarios.

This varied performance across different aspects emphasizes the importance of aligning model selection with your specific use case requirements, rather than choosing based on general performance metrics alone.

We hope you found this helpful and would love to hear from you on LinkedIn, Twitter and GitHub.

## Citation

You can cite the leaderboard with:

```bibtex
@misc{agent-leaderboard,
  author = {Pratik Bhavsar},
  title = {Agent Leaderboard},
  year = {2025},
  publisher = {Galileo.ai},
  howpublished = "\url{https://huggingface.co/spaces/galileo-ai/agent-leaderboard}"
}
```

## More Insights on Model Performance

### Reasoning Models

A notable observation from our analysis concerns reasoning-focused models. While o1 and o3-mini demonstrated excellent integration with function calling capabilities - performing at 0.876 and 0.847 respectively - we encountered significant challenges with other reasoning models. Specifically, DeepSeek V3 and Deepseek R1, despite their impressive general capabilities, were excluded from our leaderboard due to their limited function calling support.

This exclusion wasn't a reflection of model quality but rather a conscious decision based on the models' documented limitations. Both Deepseek V3 and Deepseek R1's official discussions explicitly state that function calling is not supported in their current releases. Rather than attempting to engineer workarounds or present potentially misleading performance metrics, we chose to await future releases with native function calling support.

This experience highlights that function calling is a specialized capability that shouldn't be assumed to exist in all high-performance language models. Even models with exceptional reasoning capabilities may not inherently support structured function calling without explicit design and training for this capability. It is best to evaluate the model for your use case for best selection.

### Elite Tier Performance (>= 0.9)

Gemini-2.0-flash maintains its leadership with an exceptional 0.938 average score. It demonstrates remarkable consistency across all evaluation categories, with particularly strong showings in composite scenarios (0.95) and irrelevance detection (0.98). At $0.15/$0.6 per million tokens, it offers a compelling balance of performance and cost-effectiveness.

Following closely, GPT-4o achieves 0.900, excelling in multiple tool handling (0.99) and parallel execution (0.98), though at a significantly higher price point of $2.5/$10 per million tokens.

### High Performance Segment (0.85 to 0.9)

The high-performance segment features several strong contenders. Gemini-1.5-flash maintains impressive metrics at 0.895, particularly excelling in irrelevance detection (0.98) and single function performance (0.99). Gemini-1.5-pro, despite being priced higher at $1.25/$5 per million tokens, achieves 0.885 with notably strong performance in composite tasks (0.93) and single tool execution (0.99).

o1, despite its premium pricing at $15/$60 per million tokens, justifies its position with a 0.876 score and industry-leading long-context performance (0.98). The newly included o3-mini demonstrates competitive performance at 0.847, with strong showings in single function calls (0.975) and irrelevance detection (0.97), offering a balanced option at $1.1/$4.4 per million tokens.

### Mid Tier Capabilities (0.8 to 0.85)

GPT-4o-mini maintains strong efficiency at 0.832, particularly impressive in parallel tool usage (0.99) and tool selection, though it struggles with long-context scenarios (0.51).

Among open source models, mistral-small-2501 leads at 0.832, showing remarkable improvement over its predecessor with strong long-context handling (0.92) and tool selection capabilities (0.99). Qwen-72b follows closely at 0.817, matching private models in irrelevance detection (0.99) and exhibiting strong long-context handling (0.92). Mistral-large demonstrated solid tool selection (0.97) but facing challenges in composite tasks (0.76).

Claude-sonnet achieves 0.801 with standout performance in tool miss detection (0.92) and single function handling (0.955).

### Base Tier Models (<0.8)

This tier includes models that work in specific areas despite lower overall scores. Claude-haiku offers balanced performance at 0.765 with cost-effective pricing of $0.8/$4 per million tokens.

The open-source Llama-70B shows promise at 0.774, particularly in multiple tool scenarios (0.99), while smaller variants like Mistral-small (0.750), Ministral-8b (0.689), and Mistral-nemo (0.661) provide efficient options for basic tasks.
