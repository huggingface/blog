---
title: "ScreenEnv: Deploy your full stack Desktop Agent" 
thumbnail: /blog/assets/screenenv/screenenv.png
authors:
- user: A-Mahla
- user: m-ric
---

# ScreenEnv: Deploy your full stack Desktop Agent

**TL;DR**: ScreenEnv is a powerful Python library that lets you create isolated desktop environments in Docker containers for GUI automation, testing and AI agent development. With built-in support for the Model Context Protocol (MCP) , it's never been easier to deploy desktop agents that can see, click, and interact with real applications.

## What is ScreenEnv?

---

<div style="text-align: center;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/VCutEsRSJ5A?si=PT0ETJ7zIJ9ywhGW" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

---

Imagine you need to automate desktop tasks, test GUI applications, or build an AI agent that can interact with softwares. Traditionally, this meant complex VM setups, brittle automation frameworks, or expensive cloud solutions.

ScreenEnv changes this by providing a **sandboxed desktop environment** that runs in a Docker container. Think of it as a complete virtual desktop session that your code can fully control - not just clicking buttons and typing text, but managing the entire desktop experience including launching applications, organizing windows, handling files, executing terminal commands, and recording the entire session.

## Why ScreenEnv?

- **🖥️ Full Desktop Control**: Complete mouse and keyboard automation, window management, application launching, file operations, terminal access, and screen recording
- **🤖 Dual Integration Modes**: Support both Model Context Protocol (MCP) for AI systems and direct Sandbox API - adapting to any agent or backend logic
- **🐳 Docker Native**: No complex VM setup - just Docker. The environment is isolated, reproducible, and easily deployed anywhere in less than 10 secondes. Support AMD64 and ARM64 architecture.

### 🎯 **One-Line Setup**

```python
from screenenv import Sandbox
sandbox = Sandbox()  # That's it!
```

## Two Integration Approaches

ScreenEnv provides **two complementary ways** to integrate with your agents and backend systems, giving you flexibility to choose the approach that best fits your architecture:

### Option 1: Direct Sandbox API

Perfect for custom agent frameworks, existing backends, or when you need fine-grained control:

```python
from screenenv import Sandbox

# Direct programmatic control
sandbox = Sandbox(headless=False)
sandbox.launch("xfce4-terminal")
sandbox.write("echo 'Custom agent logic'")
screenshot = sandbox.screenshot()
image = Image.open(BytesIO(screenshot_bytes))
...
sandbox.close()
# If close() isn’t called, you might need to shut down the container yourself.
```

### Option 2: MCP Server Integration

Ideal for AI systems that support the Model Context Protocol:

```python
from screenenv import MCPRemoteServer
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Start MCP server for AI integration
server = MCPRemoteServer(headless=False)
print(f"MCP Server URL: {server.server_url}")

# AI agents can now connect and control the desktop
async def mcp_session():
    async with streamablehttp_client(server.server_url) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            print(await session.list_tools())

            response = await session.call_tool("screenshot", {})
            image_bytes = base64.b64decode(response.content[0].data)
            image = Image.open(BytesIO(image_bytes))

server.close()
# If close() isn’t called, you might need to shut down the container yourself.
```

This dual approach means ScreenEnv adapts to your existing infrastructure rather than forcing you to change your agent architecture.

## ✨ Create a Desktop Agent with screenenv and smolagents

`screenenv` natively supports `smolagents`, making it easy to build your own custom Desktop Agent for automation. Here’s how to create your own AI-powered Desktop Agent in just a few steps:

### **1. Choose Your Model**

Pick the backend VLM you want to power your agent.

```python
import os

from smolagents import OpenAIServerModel
model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Inference Endpoints
from smolagents import HfApiModel
model = HfApiModel(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    token=os.getenv("HF_TOKEN"),
    provider="nebius",
)

# Transformer models
from smolagents import TransformersModel
model = TransformersModel(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Other providers
from smolagents import LiteLLMModel
model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-20250514")

# see smolagents to get the list of available model connectors

```

### **2. Define Your Custom Desktop Agent**

Inherit from `DesktopAgentBase` and implement the `_setup_desktop_tools` method to build your own action space!

```python
from screenenv import DesktopAgentBase, Sandbox
from smolagents import Model, Tool, tool
from smolagents.monitoring import LogLevel
from typing import List

class CustomDesktopAgent(DesktopAgentBase):
    """Agent for desktop automation"""

    def __init__(
        self,
        model: Model,
        data_dir: str,
        desktop: Sandbox,
        tools: List[Tool] | None = None,
        max_steps: int = 200,
        verbosity_level: LogLevel = LogLevel.INFO,
        planning_interval: int | None = None,
        use_v1_prompt: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            data_dir=data_dir,
            desktop=desktop,
            tools=tools,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval=planning_interval,
            use_v1_prompt=use_v1_prompt,
            **kwargs,
        )

        # OPTIONAL: Add a custom prompt template - see src/screenenv/desktop_agent/desktop_agent_base.py for more details about the default prompt template
        # self.prompt_templates["system_prompt"] = CUSTOM_PROMPT_TEMPLATE.replace(
        #     "<<resolution_x>>", str(self.width)
        # ).replace("<<resolution_y>>", str(self.height))
        # Important: Adjust the prompt based on your action space to improve results.

    def _setup_desktop_tools(self) -> None:
        """Define your custom tools here."""
        
        
        @tool
        def click(x: int, y: int) -> str:
            """
            Clicks at the specified coordinates.
            Args:
                x: The x-coordinate of the click
                y: The y-coordinate of the click
            """
            self.desktop.left_click(x, y)
            # self.click_coordinates = (x, y) to add the click coordinate to the observation screenshot 
            return f"Clicked at ({x}, {y})"
        
        self.tools["click"] = click
        

        @tool
        def write(text: str) -> str:
            """
            Types the specified text at the current cursor position.
            Args:
                text: The text to type
            """
            self.desktop.write(text, delay_in_ms=10)
            return f"Typed text: '{text}'"

        self.tools["write"] = write

        @tool
        def press(key: str) -> str:
            """
            Presses a keyboard key or combination of keys
            Args:
                key: The key to press (e.g. "enter", "space", "backspace", etc.) or a multiple keys string to press, for example "ctrl+a" or "ctrl+shift+a".
            """
            self.desktop.press(key)
            return f"Pressed key: {key}"

        self.tools["press"] = press
        
        @tool
        def open(file_or_url: str) -> str:
            """
            Directly opens a browser with the specified url or opens a file with the default application.
            Args:
                file_or_url: The URL or file to open
            """

            self.desktop.open(file_or_url)
            # Give it time to load
            self.logger.log(f"Opening: {file_or_url}")
            return f"Opened: {file_or_url}"

        @tool
        def launch_app(app_name: str) -> str:
            """
            Launches the specified application.
            Args:
                app_name: The name of the application to launch
            """
            self.desktop.launch(app_name)
            return f"Launched application: {app_name}"

        self.tools["launch_app"] = launch_app

        ... # Continue implementing your own action space.
```

### 3. **Run the Agent on a Desktop Task**

```python
from screenenv import Sandbox

# Define your sandbox environment
sandbox = Sandbox(headless=False, resolution=(1280, 720))

# Create your agent
agent = CustomDesktopAgent(
    model=model,
    data_dir="data",
    desktop=sandbox,
)

# Run a task
task = "Open LibreOffice, write a report of approximately 300 words on the topic ‘AI Agent Workflow in 2025’, and save the document."

result = agent.run(task)
print(f"📄 Result: {result}")

sandbox.close()
```

> 💡 For a comprehensive implementation, see this [CustomDesktopAgent](https://github.com/huggingface/screenenv/blob/main/examples/desktop_agent.py) source on GitHub.


## Get Started Today

```bash
# Install ScreenEnv
pip install screenenv

# Try the examples
git clone git@github.com:huggingface/screenenv.git
cd screenenv
python -m examples.desktop_agent
# use 'sudo -E python -m examples.desktop_agent` if you're not in 'docker' group
```

## What's Next?

ScreenEnv aims to expand beyond Linux to support **Android, macOS, and Windows**, unlocking true cross-platform GUI automation. This will enable developers and researchers to build agents that generalize across environments with minimal setup.

These advancements pave the way for creating **reproducible, sandboxed environments** ideal for benchmarking and evaluation.

Repository: https://github.com/huggingface/screenenv