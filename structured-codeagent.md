---
title: "CodeAgents + Structure: A Better Way to Execute Actions" 
thumbnail: /blog/assets/structured-codeagent/thumbnail-codeagent.png
authors:
- user: akseljoonas
- user: m-ric
---

# CodeAgents + Structure: A Better Way to Execute Actions

Today we're sharing research that bridges two powerful paradigms in AI agent design: the expressiveness of code-based actions and the reliability of structured generation. Our findings show that forcing **CodeAgents** to generate both thoughts and code in a structured JSON format can significantly outperform traditional approaches across multiple benchmarks.

![accuracy.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/structured-codeagent/accuracy.png)
Figure 1: Accuracy comparison of three approaches: Structured CodeAgent (blue), CodeAgent (orange), and ToolCallingAgent (gray) on [SmolBench (GAIA, MATH, SimpleQA, and Frames)](https://huggingface.co/datasets/akseljoonas/smolbench). Error bars represent 95% Confidence Intervals.

## **🤔 The Evolution of Agent Actions**

AI agents need to take actions in the world - whether that's calling APIs, processing data, or reasoning through complex problems. How agents express these actions has evolved through several paradigms:

**Traditional JSON Agent**: Agents generate structured JSON to call tools.

```json
{"tool": "get_weather", "arguments": {"city": "Paris"}}
```

These agents operate by selecting from a list of predefined tools and generating JSON-formatted calls. This method for calling tools has been popularized by OpenAI's [function calling API](https://openai.com/index/function-calling-and-other-api-updates/), and has since then been the most widely used method to call tools.

It is reliable, but limited by:

- **A limited set of actions**: The actions the agent can take are expressed only through predefined tools which limit its functionality.
- **Lack of composability**: If the task requires composing information from multiple sources, JSON agents struggle because they lack support for maintaining intermediate state across tool calls. While some models support parallel tool calls, they can't easily handle scenarios where one tool's output determines the next action or where results need to be compared and processed together.
- **Rigid structure**: Very limited in handling cases where tools do not match exactly what needs to be done.

**Code Agents**: Agents make use of their innate coding ability and write executable Python code directly.

```python
# We can get the average temperature in 3 cities in 1 model call.
temperature_sum = 0
for city in ["Paris", "Tokyo", "New York"]:
    temp = get_weather(city)
    temperature_sum += temp
    
print(f"Average temperature: {temperature_sum / 3:.1f}°C")
```

This shift, first presented as CodeAct in the paper “[Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030)” gave AI agents the flexibility to write arbitrary executable Python code in addition to tool-calling.

The key insight here is that **tools are called directly from within the code**, making variables and state management much more reliable. Agents can call tools within loops, functions, and conditional statements - essentially generating a dynamic graph of tool execution in each action! 

Pros of using a [CodeAgent](https://github.com/huggingface/smolagents/blob/6a12ebdf210207eec22d5940157f522463fc1c59/src/smolagents/agents.py#L1344):

- **Smart tool use**: Agents decide which tools to use based on what’s happening in the moment.
- **Unlimited flexibility**: Can use any Python functionality to achieve a goal.
- **Ability to test thoughts**: Agents can hypothesize and test, leading to more flexibility in their actions

However, parsing code from markdown can be error-prone which leads us to a proposition: why not use structured generation to generate code actions?

### **➡️ Adding Structured outputs to Code Agent**

With Structured outputs, you can force the LLM to generate explicit thoughts and code as a JSON blob:

```json
// The "code" block gets parsed into executable Python
{
  "thoughts": "I want to find the average temperature across 3 cities.",
  "code": "temperature_sum = 0\nfor city in [\"Paris\", \"Tokyo\", \"New York\"]:\n    temp = get_weather(city)\n    temperature_sum += temp\n\nprint(f\"Average temperature: {temperature_sum / 3:.1f}°C\")"
}
```

The key difference is that the generation is enforced: basically, now instead of just being prompted to output thoughts, then code, the usage of [structured outputs](https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance) forces it to respect the structure.

This approach adds the reliability of structured generation to the flexibility of code execution, thus getting the best of both worlds.

- **Explicit reasoning**: The `thoughts` field forces the agent to reason right before it takes an action.
- **Reliable parsing**: JSON structure eliminates markdown parsing errors
- **Full code expressiveness**: The `code` field maintains all the flexibility of code agents
- **Better separation**: Clear separation between planning and execution

## 🧪 **Benchmark Results**

We compared these three paradigms across multiple benchmarks including GAIA, MATH, SimpleQA, and Frames. The results show a clear pattern: **Code actions + structured generation consistently improves performance for capable models**.

Across most capable models, the structured approach consistently outperformed the regular CodeAgent approach by 2-7 percentage points on average.

- **OpenAI models**: Show the largest improvements with structure, particularly on reasoning-heavy tasks
- **Claude models**: Benefit from structure, with Claude 3.7 Sonnet showing especially strong results
- **Qwen models**: Generally improve with structure, though “structure tax” (see in next section) creeps in for smaller models.

## **💡 Why Structure (Generally) Helps**

### **The Parsing Problem is Real**

Our [implementation of CodeAgent in smolagents](https://github.com/huggingface/smolagents/blob/6a12ebdf210207eec22d5940157f522463fc1c59/src/smolagents/agents.py#L1344) extracts Python code from the LLM output, which can fail when:

- Code block formulation in markdown is incomplete or incorrectly formatted
- Multiple code blocks appear in a single response

Structured generation eliminates these issues with reliable JSON parsing.

To understand why structured generation matters, we analyzed 15,724 agent traces across our benchmarks. The results are striking:

- **2.4%** of traces had parsing errors in their first call
- Traces **with** first call parsing errors: **42.3%** success rate
- Traces **without** first call parsing errors: **51.3%** success rate

**Agent traces without parsing errors succeed 21.3% more often than those with parsing errors.**

This isn't just about convenience - parsing errors create a cascade of failures that significantly impact overall agent performance. When an agent can't execute its first action due to malformed code, it often struggles to recover, leading to suboptimal problem-solving paths.

![parsing error.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/structured-codeagent/parsing_error.png)
Figure 2: Parsing errors in the first step reduce success rates of the agent by 21.3% and increase average steps taken from 3.18 to 4.63.


**Additionally: Enforced Reasoning Process**

The use of structured generation and explicit `thoughts` not just prompts, but forces agents to articulate their reasoning before acting. This leads to:

- **Better planning**: Agents think through problems more systematically
- **Enhanced reliability**: Explicit reasoning catches logical errors early

### **The Structure Tax**

Our results also reveal a clear capability threshold: models need sufficient instruction-following ability and JSON coverage in their pre-training data to benefit from structured generation. This suggests that structured approaches work best with:

- Large, well-trained models
- Models with strong instruction-following capabilities
- Models fine-tuned on structured generation.

#### **When Structure Breaks: A Real Example**

Here's what happens when a smaller model (e.g `mistralai/Mistral-7B-Instruct-v0.3`) tries to generate structured code - the cognitive load becomes too much:

```json
{
  "thought": "I need to find the height...",
  "code": "web_search(query=\"Eiffel Tower height\")\", "
}
```

The model generates syntactically broken Python code: `web_search(query="Eiffel Tower height")",` - notice the malformed string with an extra quote and comma. This leads to an immediate SyntaxError and execution failure.

This illustrates the "structure tax": smaller models struggle to simultaneously handle JSON formatting, Python syntax, and the actual problem-solving logic. The cognitive overhead of structured generation can overwhelm models that would otherwise perform reasonably well with simpler markdown-based code generation.

## **🚀 When to Use Structured CodeAgents**

**✅ Use Structured CodeAgents when:**

- Working with capable models (32B+ parameters or frontier models)
- Tasks require complex reasoning and code execution
- You need reliable parsing of agent outputs

**⚠️ Consider alternatives when:**

- Working with smaller models that struggle with structured generation
- Simple, predefined workflows are sufficient

### How to use with smolagents:

It’s super simple! Just enable it with `use_structured_outputs_internally:`

```python
from smolagents import CodeAgent, InferenceClientModel, GoogleSearchTool

# Configure agent for structured generation
agent = CodeAgent(
    tools=[GoogleSearchTool(provider="serper")],
    model=InferenceClientModel("Qwen/Qwen3-235B-A22B", provider='nebius'),
    use_structured_outputs_internally=True # Enable structured output
)

result = agent.run("Calculate the time for a cheetah to run across the Golden Gate Bridge")

```

The LLM will generate something like this:

```json
{
  "thoughts": "I need to find the length of the Golden Gate Bridge and the top speed of a cheetah, then calculate the time.",
  "code": "bridge_info = web_search('Golden Gate Bridge length meters')\ncheetah_speed = web_search('Cheetah top speed') ..."
}

```

Then the "code" part gets executed by the agent as usual : this is the standard CodeAgent, but now it has 100% parsing reliability!

### **Implementation Tips**

1. **Clear prompting**: Ensure your prompts clearly specify the expected JSON structure
2. **Model selection**: Choose models with strong structured generation capabilities
3. **Select the right provider:** Some API providers like OpenAI or Anthropic support structured generation out of the box. If you're using Inference providers through Hugging Face,  the support of structured generation varies across providers. Here is a list of providers that support structured generation: [Structured generation support for Models in smolagents‣](https://www.notion.so/Structured-generation-support-for-Models-in-smolagents-1f51384ebcac8074a051e6dd03d1fe1d?pvs=21) 

## **The Bigger Picture - What's Next?**

This research suggests we're moving toward a more nuanced understanding of agent architectures. It's not just about "what can the agent do?" but "how should the agent think about what it's doing?"

Maybe making the reasoning process more explicit helps the model stay on track. Or maybe it's just easier to parse. Either way, it’s a win.

But this is just the beginning. There are so many questions left to explore:

- What other structural improvements could help?
- How do we make this work better across different model architectures, specifically smol models?
- What does this tell us about the nature of AI reasoning?

For now, if you're using smolagents (or building your own CodeAgent system), consider giving structured output a try. Your parsing errors will thank you, and you might just see a nice increase in performance!
