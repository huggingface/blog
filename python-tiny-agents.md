---
title: "Tiny Agents in Python: a MCP-powered agent in ~70 lines of code"
thumbnail: /blog/assets/python-tiny-agents/thumbnail.png
authors:
- user: celinah
- user: julien-c
- user: Wauplin
---
# Tiny Agents in Python: a MCP-powered agent in ~70 lines of code

Inspired by [Tiny Agents in JS](https://huggingface.co/blog/tiny-agents), we ported the idea to Python 🐍 and extended the [`huggingface_hub`](https://github.com/huggingface/huggingface_hub/) client SDK to act as a MCP Client so it can pull tools from MCP servers and pass them to the LLM during inference.
MCP ([Model Context Protocol](https://modelcontextprotocol.io/)) is an open protocol that standardizes how Large Language Models (LLMs) interact with external tools and APIs. Essentially, it gets rid of the need to write custom "glue code" for each tool, making it much simpler to plug new capabilities into your LLMs.

In this blogpost, we'll explore the implementation of a _tiny_ Agent in Python that connects to MCP servers to utilize a variety of Tools. You'll discover just how surprisingly easy it is to get your own Agent up and running.

_Spoiler_ : An Agent is essentially a while loop built right on top of an MCP Client!

## How to Run the Demo

First, you need to install the latest version of huggingface_hub with the mcp extra to get all the necessary components.

```bash
pip install "huggingface_hub[mcp]==0.32.0"
```

Now, you can run an agent using the CLI. The coolest part? You can load agents directly from the Hugging Face Hub [tiny-agents](https://huggingface.co/datasets/tiny-agents/tiny-agents) Dataset or specify a path to your own local agent configuration! 

```bash
> tiny-agents run --help
                                                                                                                                      
 Usage: tiny-agents run [OPTIONS] [PATH] COMMAND [ARGS]...                                                                            
                                                                                                                                      
 Run the Agent in the CLI                                                                                                             
                                                                                                                                      
                                                                                                                                      
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   path      [PATH]  Path to a local folder containing an agent.json file or a built-in agent stored in a Hugging Face dataset      │
│                     (default: https://huggingface.co/datasets/tiny-agents/tiny-agents)                                             │
│                     [default: None]                                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```
The following command loads an agent configured to use [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) model via Nebius inference provider, and it comes equipped with a playwright MCP server, enabling it to perform browser automation tasks. The agent config is loaded from the [tiny-agents/tiny-agents](https://huggingface.co/datasets/tiny-agents/tiny-agents) Hugging Face dataset 

```bash
> tiny-agents run julien-c/local-coder
```
**TODO:** add video demo and snippet above

When you run an agent, you'll see it load, listing the tools it has discovered from its connected MCP servers. Then, it's ready for your prompts!

### Agent Configuration 

Each agent's behavior (its default model, inference provider, which MCP servers to connect to, and its initial system prompt) is defined by an `agent.json` file. You can also provide a custom `PROMPT.md` in the same directory for a more detailed system prompt. Here is an example:

`agent.json`
```json

{
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "nebius",
    "servers": [
        {
            "type": "http", 
            // An "http" MCP server is a web service that exposes Tools for the Agent.
            // The Agent communicates with it over HTTP/HTTPS using the URL below
            // to discover and execute those tools.
            "config": {
                "url": "https://evalstate-hf-mcp-server.hf.space/mcp"
            }
        }
    ]
}

```
`PROMPT.md`

```
You are an agent - please keep going until the user’s query is completely resolved [...]

```


> [!TIP]
> You can find more details about Hugging Face Inference Providers [here](https://huggingface.co/docs/inference-providers/index).

### LLMs Can Use Tools

modern LLMs are built for function calling (or tool use), which enables users to easily build applications tailored to specific use cases and real-world tasks. 

A function is defined by its schema, which informs the LLM what it does and what input arguments it expects. The LLM decides when to use a tool, the Agent then orchestrates running the tool and feeding the result back.

```python
tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Paris, France"
                        }
                    },
                    "required": ["location"],
                },
            }
        }
]
```

`InferenceClient` implements the same tool calling interface as the [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/function-calling?api-mode=chat), which is the established standard for inference providers and the community

## Building our Python MCPClient (Now in `huggingface_hub`!)

The `MCPClient` is the heart of our tool-use functionality. It's now part of `huggingface_hub` and uses the `AsyncInferenceClient` to communicate with LLMs.

> [!TIP]
> The full `MCPClient` code is in [here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/mcp_client.py) if you want to follow along using the actual code 🤓

Key responsibilities of the `MCPClient`:

- Manage async connections to one or more MCP servers.
- Discover tools from these servers.
- Format these tools for the LLM.
- Execute tool calls via the correct MCP server.

​​Here’s a glimpse of how it connects to an MCP server (the `add_mcp_server` method):

```python
class MCPClient:
    ...
    async def add_mcp_server(self, type: ServerType, **params: Any):
        # 'type' can be "stdio", "sse", or "http"
        # 'params' are specific to the server type, e.g.:
        # for "stdio": {"command": "my_tool_server_cmd", "args": ["--port", "1234"]}
        # for "http": {"url": "http://my.tool.server/mcp"}

        # 1. Establish connection based on type (stdio, sse, http)
        #    (Uses mcp.client.stdio_client, sse_client, or streamablehttp_client)
        read, write = await self.exit_stack.enter_async_context(...)

        # 2. Create an MCP ClientSession
        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream=read, write_stream=write, ...)
        )
        await session.initialize()

        # 3. List tools from the server
        response = await session.list_tools()
        for tool in response.tools:
            # Store session for this tool
            self.sessions[tool.name] = session 
            #  Add tool to the list of available tools and Format for LLM
            self.available_tools.append({ 
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })

```
It supports `stdio` servers for local tools (like accessing your file system), and `sse`/`http` servers for remote tools!

## Using the Tools: Streaming and Processing

The `MCPClient`'s `process_single_turn_with_tools` method is where the LLM interaction happens. It sends the conversation history and available tools to the LLM via `AsyncInferenceClient.chat.completions.create(..., stream=True)`.

### 1. Prepare tools and calling the LLM

First, the method determines all tools the LLM should be aware of for the current turn – this includes tools from MCP servers and any special "exit loop" tools for agent control, then, it makes a streaming call to the LLM

```python
# From MCPClient.process_single_turn_with_tools

    # Prepare tools list based on options
    tools = self.available_tools
    if exit_loop_tools is not None:
        tools = [*exit_loop_tools, *self.available_tools]

    # Create the streaming request to the LLM
    response = await self.client.chat.completions.create(
        messages=messages,
        tools=tools,
        tool_choice="auto",  # LLM decides if it needs a tool
        stream=True,  
    )

```

As chunks arrive from the LLM, the method iterates through them. Each chunk is immediately yielded, then we reconstruct the complete text response and any tool calls.

```python
# Lines 242-279 of `MCPClient.process_single_turn_with_tools`
# Read from stream
async for chunk in response:
      # Yield each chunk to caller
      yield chunk
      # Aggregate LLM's text response and parts of tool calls
      …
```

### 2. Executing tools

Once the stream is complete, if the LLM requested any tool calls (now fully reconstructed in final_tool_calls), the method processes each one:

```python
# From MCPClient.process_single_turn_with_tools
for tool_call in final_tool_calls.values():
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")

    # Prepare a message to store the tool's result
    tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": "", "name": function_name}

    # a. Is this a special "exit loop" tool?
    if exit_loop_tools and function_name in [t.function.name for t in exit_loop_tools]:
        # If so, yield a message and terminate this turn's processing
        messages.append(ChatCompletionInputMessage.parse_obj_as_instance(tool_message))
        yield ChatCompletionInputMessage.parse_obj_as_instance(tool_message)
        return # The Agent's main loop will handle this signal

    # b. It's a regular tool: find the MCP session and execute it
    session = self.sessions.get(function_name) # self.sessions maps tool names to MCP connections
    if session is not None:
        result = await session.call_tool(function_name, function_args)
        tool_message["content"] = format_result(result) # format_result processes tool output
    else:
        tool_message["content"] = f"Error: No session found for tool: {function_name}"

    # Add tool result to history and yield it
    ...

```

here, it first check if the tool called is a special control signal (an `exit_loop_tool`). If not, it finds the correct MCP session responsible for that tool and calls `session.call_tool()`. The result (it can be an error as well) is then formatted, added to the conversation history, and yielded so the Agent is aware of the tool's output.

## Our Tiny Python Agent: It's (Almost) Just a Loop! 

With the MCPClient doing all the job for tool interactions, our Agent class becomes wonderfully simple. It inherits from MCPClient and adds the conversation management logic.

> [!TIP]
> The Agent class is tiny and focuses on the conversational loop, the code can be found in https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/agent.py

### 1. Intializing the Agent

When an Agent is created, it takes an agent config (model, provider, which MCP servers to use, system prompt) and initializes the conversation history with the system prompt. The `load_tools()` method then iterates through the server configurations (defined in agent.json) and calls `add_mcp_server` (from the parent `MCPClient`) for each one, populating the agent's toolbox.

```python
# From your Agent class in agent.py

class Agent(MCPClient):
    def __init__(
        self,
        *,
        model: str,
        servers: Iterable[Dict], # Configuration for MCP servers
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None, # The system prompt
    ):
        # Initialize the underlying MCPClient with model, provider, etc.
        super().__init__(model=model, provider=provider, api_key=api_key)
        # Store server configurations to be loaded
        self._servers_cfg = list(servers)
        # Start the conversation with a system message
        self.messages: List[Union[Dict, ChatCompletionInputMessage]] = [
            {"role": "system", "content": prompt or DEFAULT_SYSTEM_PROMPT}
        ]

    async def load_tools(self) -> None:
        # Connect to all configured MCP servers and register their tools
        for cfg in self._servers_cfg:
            await self.add_mcp_server(cfg["type"], **cfg["config"])

```

### 2. The agent’s core: the Loop

The `Agent.run()` method is an asynchronous generator that processes a single user input. It manages the conversation turns, deciding when the agent's current task is complete.

```python
# Key parts of Agent.run() in agent.py
async def run(self, user_input: str, *, abort_event: Optional[asyncio.Event] = None, ...) -> AsyncGenerator[...]:
    ...
    while True: # Main loop for processing the user_input
        ...

        # Delegate to MCPClient to interact with LLM and tools for one step.
        # This streams back LLM text, tool call info, and tool results.
        async for item in self.process_single_turn_with_tools(
            self.messages,
            ...
        ):
            yield item 

        ... 
        
        # Exit Conditions
        # 1. Was an "exit" tool  called?
        if last.get("role") == "tool" and last.get("name") in {t.function.name for t in EXIT_LOOP_TOOLS}:
                return

        # 2. Max turns reached or LLM gave a final text answer?
        if last.get("role") != "tool" and num_turns > MAX_NUM_TURNS:
                return
        if last.get("role") != "tool" and next_turn_should_call_tools:
            return
        
        next_turn_should_call_tools = (last_message.get("role") != "tool")
```

Inside the `run()` loop:
- It first adds the user prompt to the conversation.
- Then it calls crucial `MCPClient.process_single_turn_with_tools(...)` to get the LLM's response and handle any tool executions for one step of reasoning.
- Each item is immediately yielded, enabling real-time streaming to the caller.
- After each step, it checks exit conditions: if a special “exit loop” tools was used, if a maximum turn limit is hit, or if the LLM provides a text response that seems final for the current request.

## Next Steps
