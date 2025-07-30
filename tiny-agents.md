---
title: "Tiny Agents: an MCP-powered agent in 50 lines of code"
thumbnail: /blog/assets/tiny-agents/thumbnail.jpg
authors:
- user: julien-c
---

# Tiny Agents: an MCP-powered agent in 50 lines of code

> [!WARNING]
> New! (May 23, '25) If you prefer Python, check out the companion post
> [`Tiny Agents in Python`](https://huggingface.co/blog/python-tiny-agents).

Over the past few weeks, I've been diving into MCP ([Model Context Protocol](https://modelcontextprotocol.io/)) to understand what the hype around it was all about.

My TL;DR is that it's fairly simple, but still quite powerful: **MCP is a standard API to expose sets of Tools that can be hooked to LLMs.**

It is fairly simple to extend an Inference Client – at HF, we have two official client SDKs: [`@huggingface/inference`](https://github.com/huggingface/huggingface.js) in JS, and [`huggingface_hub`](https://github.com/huggingface/huggingface_hub/) in Python – to also act as a MCP client and hook the available tools from MCP servers into the LLM inference. 

But while doing that, came my second realization:

> [!TIP]
> **Once you have an MCP Client, an Agent is literally just a while loop on top of it.**

In this short article, I will walk you through how I implemented it in Typescript (JS), how you can adopt MCP too and how it's going to make Agentic AI way simpler going forward.

![meme](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tiny-agents/thumbnail.jpg)
<figcaption>Image credit https://x.com/adamdotdev</figcaption>

## How to run the complete demo

If you have NodeJS (with `pnpm` or `npm`), just run this in a terminal:

```bash
npx @huggingface/mcp-client
```

or if using `pnpm`:

```bash
pnpx @huggingface/mcp-client
```

This installs my package into a temporary folder then executes its command.

You'll see your simple Agent connect to two distinct MCP servers (running locally), loading their tools, then prompting you for a conversation.

<video controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tiny-agents/use-filesystem.mp4" type="video/mp4">
</video>

By default our example Agent connects to the following two MCP servers:

- the "canonical" [file system server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem), which gets access to your Desktop,
- and the [Playwright MCP](https://github.com/microsoft/playwright-mcp) server, which knows how to use a sandboxed Chromium browser for you.

> [!NOTE]
> Note: this is a bit counter-intuitive but currently, all MCP servers are actually local processes (though remote servers are coming soon).

Our input for this first video was:

> write a haiku about the Hugging Face community and write it to a file named "hf.txt" on my Desktop

Now let us try this prompt that involves some Web browsing:

> do a Web Search for HF inference providers on Brave Search and open the first 3 results

<video controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/tiny-agents/brave-search.mp4" type="video/mp4">
</video>

### Default model and provider

In terms of model/provider pair, our example Agent uses by default:
- ["Qwen/Qwen2.5-72B-Instruct"](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- running on [Nebius](https://huggingface.co/docs/inference-providers/providers/nebius)

This is all configurable through env variables! See:

```ts
const agent = new Agent({
	provider: process.env.PROVIDER ?? "nebius",
	model: process.env.MODEL_ID ?? "Qwen/Qwen2.5-72B-Instruct",
	apiKey: process.env.HF_TOKEN,
	servers: SERVERS,
});
```

## Where does the code live

The Tiny Agent code lives in the `mcp-client` sub-package of the `huggingface.js` mono-repo, which is the GitHub mono-repo in which all our JS libraries reside.

https://github.com/huggingface/huggingface.js/tree/main/packages/mcp-client

> [!TIP]
> The codebase uses modern JS features (notably, async generators) which make things way easier to implement, especially asynchronous events like the LLM responses. 
> You might need to ask a LLM about those JS features if you're not yet familiar with them.


## The foundation for this: tool calling native support in LLMs.

What is going to make this whole blogpost very easy is that the recent crop of LLMs (both closed and open) have been trained for function calling, aka. tool use.

A tool is defined by its name, a description, and a JSONSchema representation of its parameters.
In some sense, it is an opaque representation of any function's interface, as seen from the outside (meaning, the LLM does not care how the function is actually implemented).

```ts
const weatherTool = {
	type: "function",
	function: {
		name: "get_weather",
		description: "Get current temperature for a given location.",
		parameters: {
			type: "object",
			properties: {
				location: {
					type: "string",
					description: "City and country e.g. Bogotá, Colombia",
				},
			},
		},
	},
};
```

The canonical documentation I will link to here is [OpenAI's function calling doc](https://platform.openai.com/docs/guides/function-calling?api-mode=chat). (Yes... OpenAI pretty much defines the LLM standards for the whole community 😅).

Inference engines let you pass a list of tools when calling the LLM, and the LLM is free to call zero, one or more of those tools.
As a developer, you run the tools and feed their result back into the LLM to continue the generation.

> [!NOTE]
> Note that in the backend (at the inference engine level), the tools are simply passed to the model in a specially-formatted `chat_template`, like any other message, and then parsed out of the response (using model-specific special tokens) to expose them as tool calls. See an example [in our chat-template playground](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen/Qwen3-235B-A22B&example=tool-usage).

## Implementing an MCP client on top of InferenceClient

Now that we know what a tool is in recent LLMs, let us implement the actual MCP client.

The official doc at https://modelcontextprotocol.io/quickstart/client is fairly well-written. You only have to replace any mention of the Anthropic client SDK by any other OpenAI-compatible client SDK. (There is also a [llms.txt](https://modelcontextprotocol.io/llms-full.txt) you can feed into your LLM of choice to help you code along). 

As a reminder, we use HF's `InferenceClient` for our inference client.

> [!TIP]
> The complete `McpClient.ts` code file is [here](https://github.com/huggingface/huggingface.js/blob/main/packages/mcp-client/src/McpClient.ts) if you want to follow along using the actual code 🤓

Our `McpClient` class has:
- an Inference Client (works with any Inference Provider, and `huggingface/inference` supports both remote and local endpoints)
- a set of MCP client sessions, one for each connected MCP server (yes, we want to support multiple servers)
- and a list of available tools that is going to be filled from the connected servers and just slightly re-formatted.

```ts
export class McpClient {
	protected client: InferenceClient;
	protected provider: string;
	protected model: string;
	private clients: Map<ToolName, Client> = new Map();
	public readonly availableTools: ChatCompletionInputTool[] = [];

	constructor({ provider, model, apiKey }: { provider: InferenceProvider; model: string; apiKey: string }) {
		this.client = new InferenceClient(apiKey);
		this.provider = provider;
		this.model = model;
	}
	
	// [...]
}
```

To connect to a MCP server, the official `@modelcontextprotocol/sdk/client` TypeScript SDK provides a `Client` class with a `listTools()` method:

```ts
async addMcpServer(server: StdioServerParameters): Promise<void> {
	const transport = new StdioClientTransport({
		...server,
		env: { ...server.env, PATH: process.env.PATH ?? "" },
	});
	const mcp = new Client({ name: "@huggingface/mcp-client", version: packageVersion });
	await mcp.connect(transport);

	const toolsResult = await mcp.listTools();
	debug(
		"Connected to server with tools:",
		toolsResult.tools.map(({ name }) => name)
	);

	for (const tool of toolsResult.tools) {
		this.clients.set(tool.name, mcp);
	}

	this.availableTools.push(
		...toolsResult.tools.map((tool) => {
			return {
				type: "function",
				function: {
					name: tool.name,
					description: tool.description,
					parameters: tool.inputSchema,
				},
			} satisfies ChatCompletionInputTool;
		})
	);
}
```

`StdioServerParameters` is an interface from the MCP SDK that will let you easily spawn a local process: as we mentioned earlier, currently, all MCP servers are actually local processes.

For each MCP server we connect to, we slightly re-format its list of tools and add them to `this.availableTools`.

### How to use the tools

Easy, you just pass `this.availableTools` to your LLM chat-completion, in addition to your usual array of messages:

```ts
const stream = this.client.chatCompletionStream({
	provider: this.provider,
	model: this.model,
	messages,
	tools: this.availableTools,
	tool_choice: "auto",
});
```

`tool_choice: "auto"` is the parameter you pass for the LLM to generate zero, one, or multiple tool calls.

When parsing or streaming the output, the LLM will generate some tool calls (i.e. a function name, and some JSON-encoded arguments), which you (as a developer) need to compute. The MCP client SDK once again makes that very easy; it has a `client.callTool()` method:

```ts
const toolName = toolCall.function.name;
const toolArgs = JSON.parse(toolCall.function.arguments);

const toolMessage: ChatCompletionInputMessageTool = {
	role: "tool",
	tool_call_id: toolCall.id,
	content: "",
	name: toolName,
};

/// Get the appropriate session for this tool
const client = this.clients.get(toolName);
if (client) {
	const result = await client.callTool({ name: toolName, arguments: toolArgs });
	toolMessage.content = result.content[0].text;
} else {
	toolMessage.content = `Error: No session found for tool: ${toolName}`;
}
```

Finally you will add the resulting tool message to your `messages` array and back into the LLM.

## Our 50-lines-of-code Agent 🤯

Now that we have an MCP client capable of connecting to arbitrary MCP servers to get lists of tools and capable of injecting them and parsing them from the LLM inference, well... what is an Agent?

> Once you have an inference client with a set of tools, then an Agent is just a while loop on top of it.

In more detail, an Agent is simply a combination of:
- a system prompt
- an LLM Inference client
- an MCP client to hook a set of Tools into it from a bunch of MCP servers
- some basic control flow (see below for the while loop)

> [!TIP]
> The complete `Agent.ts` code file is [here](https://github.com/huggingface/huggingface.js/blob/main/packages/mcp-client/src/Agent.ts).

Our Agent class simply extends McpClient:

```ts
export class Agent extends McpClient {
	private readonly servers: StdioServerParameters[];
	protected messages: ChatCompletionInputMessage[];

	constructor({
		provider,
		model,
		apiKey,
		servers,
		prompt,
	}: {
		provider: InferenceProvider;
		model: string;
		apiKey: string;
		servers: StdioServerParameters[];
		prompt?: string;
	}) {
		super({ provider, model, apiKey });
		this.servers = servers;
		this.messages = [
			{
				role: "system",
				content: prompt ?? DEFAULT_SYSTEM_PROMPT,
			},
		];
	}
}
```

By default, we use a very simple system prompt inspired by the one shared in the [GPT-4.1 prompting guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide).

Even though this comes from OpenAI 😈, this sentence in particular applies to more and more models, both closed and open:

> We encourage developers to exclusively use the tools field to pass tools, rather than manually injecting tool descriptions into your prompt and writing a separate parser for tool calls, as some have reported doing in the past.

Which is to say, we don't need to provide painstakingly formatted lists of tool use examples in the prompt. The `tools: this.availableTools` param is enough.

Loading the tools on the Agent is literally just connecting to the MCP servers we want (in parallel because it's so easy to do in JS):

```ts
async loadTools(): Promise<void> {
	await Promise.all(this.servers.map((s) => this.addMcpServer(s)));
}
```

We add two extra tools (outside of MCP) that can be used by the LLM for our Agent's control flow:

```ts
const taskCompletionTool: ChatCompletionInputTool = {
	type: "function",
	function: {
		name: "task_complete",
		description: "Call this tool when the task given by the user is complete",
		parameters: {
			type: "object",
			properties: {},
		},
	},
};
const askQuestionTool: ChatCompletionInputTool = {
	type: "function",
	function: {
		name: "ask_question",
		description: "Ask a question to the user to get more info required to solve or clarify their problem.",
		parameters: {
			type: "object",
			properties: {},
		},
	},
};
const exitLoopTools = [taskCompletionTool, askQuestionTool];
```

When calling any of these tools, the Agent will break its loop and give control back to the user for new input.

### The complete while loop

Behold our complete while loop.🎉

The gist of our Agent's main while loop is that we simply iterate with the LLM alternating between tool calling and feeding it the tool results, and we do so **until the LLM starts to respond with two non-tool messages in a row**.

This is the complete while loop:

```ts
let numOfTurns = 0;
let nextTurnShouldCallTools = true;
while (true) {
	try {
		yield* this.processSingleTurnWithTools(this.messages, {
			exitLoopTools,
			exitIfFirstChunkNoTool: numOfTurns > 0 && nextTurnShouldCallTools,
			abortSignal: opts.abortSignal,
		});
	} catch (err) {
		if (err instanceof Error && err.message === "AbortError") {
			return;
		}
		throw err;
	}
	numOfTurns++;
	const currentLast = this.messages.at(-1)!;
	if (
		currentLast.role === "tool" &&
		currentLast.name &&
		exitLoopTools.map((t) => t.function.name).includes(currentLast.name)
	) {
		return;
	}
	if (currentLast.role !== "tool" && numOfTurns > MAX_NUM_TURNS) {
		return;
	}
	if (currentLast.role !== "tool" && nextTurnShouldCallTools) {
		return;
	}
	if (currentLast.role === "tool") {
		nextTurnShouldCallTools = false;
	} else {
		nextTurnShouldCallTools = true;
	}
}
```

## Next steps

There are many cool potential next steps once you have a running MCP Client and a simple way to build Agents 🔥

- Experiment with **other models**
  - [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) is optimized for function calling
  - Gemma 3 27B, the [Gemma 3 QAT](https://huggingface.co/collections/google/gemma-3-qat-67ee61ccacbf2be4195c265b) models are a popular choice for function calling though it would require us to implement tool parsing as it's not using native `tools` (a PR would be welcome!)
- Experiment with all the **[Inference Providers](https://huggingface.co/docs/inference-providers/index)**:
  - Cerebras, Cohere, Fal, Fireworks, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, etc.
  - each of them has different optimizations for function calling (also depending on the model) so performance may vary!
- Hook **local LLMs** using llama.cpp or LM Studio


Pull requests and contributions are welcome! 
Again, everything here is [open source](https://github.com/huggingface/huggingface.js)! 💎❤️



