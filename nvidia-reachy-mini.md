---
title: "NVIDIA brings agents to life with DGX Spark and Reachy Mini" 
thumbnail: /blog/assets/nvidia-reachy-mini/nvidia-reachy-mini-compressed.png
authors:
- user: jeffboudier
- user: nader-at-nvidia
  guest: true
  org: nvidia
- user: alecfong
  guest: true
  org: nvidia
---

# NVIDIA brings agents to life with DGX Spark and Reachy Mini

![NVIDIA creates real world agents with DGX Spark and Reachy Mini at the CES 2026 Keynote](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nvidia-reachy-mini/nvidia-reachy-mini-compressed.png)

Today at CES 2026, NVIDIA unveiled a world of new open models to enable the future of agents, online and in the real world. From the recently released [NVIDIA Nemotron](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) reasoning LLMs to the new [NVIDIA Isaac GR00T N1.6](https://huggingface.co/nvidia/GR00T-N1.6-3B) open reasoning VLA and [NVIDIA Cosmos world foundation models](https://huggingface.co/collections/nvidia/cosmos-reason2), all the building blocks are here today for AI Builders to build their own agents.

But what if you could bring your own agent to life, right at your desk? An AI buddy that can be useful to you and process your data privately?

In the CES keynote today, Jensen Huang showed us how we can do exactly that, using the processing power of NVIDIA [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) with [Reachy Mini](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini) to create your own little office R2D2 you can talk to and collaborate with.

This blog post provides a step-by-step guide to replicate this amazing experience at home using a DGX Spark and [Reachy Mini](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini).

![Reachy Mini and DGX Spark on a desk](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nvidia-reachy-mini/reachy-mini-dgx-spark.png)

Let’s dive in!

## Ingredients

If you want to start cooking right away, here’s the [source code of the demo](https://github.com/brevdev/reachy-personal-assistant). 

We’ll be using the following:

1. A reasoning model: demo uses [NVIDIA Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)  
2. A vision model: demo uses [NVIDIA Nemotron Nano 2 VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)  
3. A text-to-speech model: demo uses [ElevenLabs](https://elevenlabs.io)  
4. [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) (or [Reachy Mini Simulation](https://github.com/pollen-robotics/reachy_mini/blob/develop/docs/platforms/simulation/get_started.md))  
5. Python v3.10+ environment, with [uv](https://docs.astral.sh/uv/)

Feel free to adapt the recipe and make it your own \- you have many ways to integrate the models into your application:

1. Local deployment – Run on your own hardware ([DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) or a GPU with sufficient VRAM). Our implementation requires \~65GB disk space for the reasoning model, and \~28GB for the vision model.  
2. Cloud deployment– Deploy the models on cloud GPUs e.g. through [NVIDIA Brev](http://build.nvidia.com/gpu) or [Hugging Face Inference Endpoints](https://endpoints.huggingface.co/).  
3. Serverless model endpoints – Send requests to [NVIDIA](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/deploy) or [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index).

## Giving agentic powers to Reachy

Turning an AI agent from a simple chat interface into something you can interact with naturally makes conversations feel more real. When an AI agent can see through a camera, speak out loud, and perform actions, the experience becomes more engaging. That’s what Reachy Mini makes possible.

Reachy Mini is designed to be customizable. With access to sensors, actuators, and APIs, you can easily wire it into your existing agent stack, by simulation or real hardware controlled directly from Python. 

This post focuses on composing existing building blocks rather than reinventing them. We combine open models for reasoning and vision, an agent framework for orchestration, and tool handlers for actions. Each component is loosely coupled, making it easy to swap models, change routing logic, or add new behaviors.

Unlike closed personal assistants, this setup stays fully open. You control the models, the prompts, the tools, and the robot’s actions. Reachy Mini simply becomes the physical endpoint of your agent where perception, reasoning, and action come together.

## Building the agent

In this example, we use the NVIDIA [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit), a flexible, lightweight, framework-agnostic open source library, to connect all the components of the agent together. It works seamlessly with other agentic frameworks, like LangChain, LangGraph, CrewAI, handling how models interact, routing inputs and outputs between them, and making it easy to experiment with different configurations or add new capabilities without rewriting core logic. The toolkit also provides built-in profiling and optimization features, letting you track token usage efficiency and latency across tools and agents, identify bottlenecks, and automatically tune hyperparameters to maximize accuracy while reducing cost and latency.

## Step 0: Set up and get access to models and services

First, clone the repository that contains all the code you’ll need to follow along:

```shell
git clone git@github.com/brevdev/reachy-personal-assistant
cd reachy-personal-assistant
```

To access your intelligence layer, powered by the NVIDIA Nemotron models, you can either deploy them using [NVIDIA NIM](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html) or [vLLM](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16#use-it-with-vllm), or connect to them through remote endpoints available at [build.nvidia.com](http://build.nvidia.com).

The following instructions assume you are accessing the Nemotron models via endpoints. Create a .env file in the main directory with your API keys. For local deployments, you do not need to specify API keys and can skip this step.

```shell
NVIDIA_API_KEY=your_nvidia_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

## Step 1: Build a chat interface

Start by getting a basic LLM chat workflow running through NeMo Agent Toolkit’s API server. NeMo Agent Toolkit supports running workflows via \`nat serve\` and providing a config file. The [config file](https://github.com/brevdev/reachy-personal-assistant/blob/main/nat/src/ces_tutorial/config.yml) passed here contains all the necessary setup information for the agent, which includes the models used for chat, image understanding, as well as the router model used by the agent. The [NeMo Agent Toolkit UI](https://github.com/NVIDIA/NeMo-Agent-Toolkit-UI) can connect over HTTP/WebSocket so you can chat with your workflow like a standard chat product. In this implementation, the NeMo Agent Toolkit server is launched on port 8001 (so your bot can call it, and the UI can too): 

```shell
cd nat
uv venv
uv sync
uv run --env-file ../.env nat serve --config_file src/ces_tutorial/config.yml --port 8001
```

Next, verify that you can send a plain text prompt through a separate terminal to ensure everything is setup correctly:

```shell
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

Reviewing the agent configuration, you’ll notice it defines far more capabilities than a simple chat completion. The next steps will walk through those details.

## Step 2: Add NeMo Agent Toolkit’s built-in ReAct agent for tool calling

Tool calling is an essential part of AI agents. NeMo Agent Toolkit includes a built-in ReAct agent that can reason between tool calls and use multiple tools before answering. We route “action requests” to a ReAct agent that’s allowed to call tools (for example, tools that trigger robot behaviors or fetch current robot state). 

Some practical notes to keep in mind:

* Keep tool schemas tight (clear name/description/args), because that’s what the agent uses to decide what to call.   
* Put a hard cap on steps (max\_tool\_calls) so the agent can’t spiral.   
* If using a physical robot, consider a “confirm before actuation” pattern for physical actions to ensure movement safety.

Take a look at this portion of the config it defines the tools (like Wikipedia search) and specifies the ReAct agent pattern used to manage them.

```xml
functions:
   wikipedia_search:
      _type: wiki_search
      max_results: 2
      
   ..
   react_agent:
      _type: react_agent
      llm_name: agent_llm
      verbose: true
      parse_agent_response_max_retries: 3
      tool_names: [wikipedia_search]

​​workflow:
   _type: ces_tutorial_router_agent
   agent: react_agent
```

## Step 3: Add a router to direct queries to different models

The key idea: don’t use one model for everything. Instead, route based on intent:

* Text queries can use a fast text model  
* Visual queries must be run through a VLM  
* Action/tool requests are routed to the ReAct agent \+ tools 

You can implement routing a few ways (heuristics, a lightweight classifier, or a dedicated routing service). If you want the “production” version of this idea, the NVIDIA [LLM Router developer example](https://build.nvidia.com/nvidia/llm-router) is the full reference implementation and includes evaluation and monitoring patterns. 

A basic routing policy might work like this:

* If the user is asking a question about their environment, then send the request to a VLM along with an image captured from the camera (or Reachy).  
* If the user asks a question requiring real time information, send the input to a ReACT agent to perform a web search via a tool call.  
* If the user is asking simple questions, send the request to a small and fast model optimized for chit chat.

These sections of the config define the routing topology and specify the router model.

```xml
functions:
   ..
   
   router:
      _type: router
      route_config:
        - name: other
          description: Any question that requires careful thought, outside information, image understanding, or tool calling to take actions.
        - name: chit_chat
          description: Any simple chit chat, small talk, or casual conversation.
        - name: image_understanding
          description: A question that requires the assistant to see the user eg a question about their appearance, environment, scene or surroundings. Examples what am I holding, what am I wearing, what do I look like, what is in my surroundings, what does it say on the whiteboard. Questions about attire eg what color is my shirt/hat/jacket/etc
      llm_name: routing_llm


llms:
   ..
   routing_llm: 
      _type: nim
      model_name: microsoft/phi-3-mini-128k-instruct
      temperature: 0.0
```

### 

**NOTE**: If you want to reduce latency/cost or run offline, you can self-host one of the routed models (typically the “fast text” model) and keep the VLM remote. One common approach is serving via NVIDIA NIM or vLLM and pointing NeMo Agent Toolkit to an OpenAI-compatible endpoint. 

## Step 4: Add a Pipecat bot for real-time voice \+ vision 

Now we go real time. [Pipecat](https://www.pipecat.ai/) is a framework designed for low-latency voice/multimodal agents: it orchestrates audio/video streams, AI services, and transports so you can build natural conversations. In this repo, the bot service is responsible for:

* Capturing vision (robot camera)  
* Speech recognition \+ text-to-speech  
* Coordinating robot movement and expressive behaviors 

You will find all the pipecat bot code in the \`reachy-personal-assistant/bot\` [folder](https://github.com/brevdev/reachy-personal-assistant/tree/main/bot).

## Step 5: Hook everything up to Reachy (hardware or simulation)

Reachy Mini exposes a daemon that the rest of your system connects to. The repo runs the daemon in simulation by default (--sim). If you have access to a real Reachy you can remove this flag and the same code will control your robot.

### Run the full system

You will need three terminals to run the entire system:

#### Terminal 1: Reachy daemon

```
cd bot
# macOS:
uv run mjpython -m reachy_mini.daemon.app.main --sim --no-localhost-only

# Linux:
uv run -m reachy_mini.daemon.app.main --sim --no-localhost-only
```

If you are using the physical hardware, remember to omit the \--sim flag from the command.

#### Terminal 2: Bot service

```
cd bot
uv venv
uv sync
uv run --env-file ../.env python main.py
```

#### Terminal 3: NeMo Agent Toolkit service

If the NeMo Agent Toolkit service is not already running from Step 1, start it now in Terminal 3\.

```
cd nat
uv venv
uv sync
uv run --env-file ../.env nat serve --config_file src/ces_tutorial/config.yml --port 8001
```

## 

Once all the terminals are set up, there are two main windows to keep track of:

1. **Reachy Sim** – This window appears automatically when you start the simulator daemon in Terminal 1\. This is applicable if you’re running Reachy mini simulation in place of the physical device.  
     
2. **Pipecat Playground** – This is the client-side UI where you can connect to the agent, enable microphone and camera inputs, and view live transcripts. In Terminal 2, open the URL exposed by the bot service: [http://localhost:7860/](http://localhost:7860/client/). Click “CONNECT” in your browser. It may take a few seconds to initialize, and you’ll be prompted to grant microphone (and optionally camera) access.  
     
     
   Once both windows are up and running:   
* The Client and Agent STATUS indicators should show READY  
* The bot will greet you with a welcome message “Hello, how may I assist you today?”


At this point, you can start interacting with your agent\!

## Try these example prompts

Here are a few simple prompts to help you test your personal assistant. You can start with these and then experiment by adding your own to see how the agent responds\!

**Text-only prompts (routes to the fast text model)**

* “Explain what you can do in one sentence.”  
* “Summarize the last thing I said.”

**Vision prompts (routes to the VLM)**

* “What am I holding up to the camera?”  
* “Read the text on this page and summarize it.”

## Where to go next

Instead of a "black-box" assistant, this builds a foundation for a private, hackable system where you can control both the intelligence and the hardware. You can inspect, extend, and run it locally, with full visibility into data flow, tool permissions, and how the robot perceives and acts.

Depending on your goals, here are a few directions to explore next:

* Optimize for performance: Use the [LLM Router developer example](https://github.com/NVIDIA-AI-Blueprints/llm-router/tree/experimental) to balance cost, latency, and quality by intelligently directing queries between different models.  
* Check out the tutorial for building a voice-powered RAG agent with guardrails using Nemotron open models.   
* Master the hardware: Explore the Reachy Mini SDK and simulation docs to design and test advanced robotic behaviors before deploying to your physical system.  
* Explore and contribute to the [apps](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini#apps) built by the community for Reachy.

Want to try it right away? Deploy [the full environment here](https://brev.nvidia.com/launchable/deploy?launchableID=env-37ZZY950tDIuCQSSHXIKSGpRbFJ). One click and you're running.
