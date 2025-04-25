

# Tiny Agents: a MCP-powered agent in 50 lines of code

Over the past few weeks, I've been diving into MCP ([Model Context Protocol](https://modelcontextprotocol.io/)) to understand what the hype around it was all about.

My TL;DR is that it's fairly simple, but still quite powerful: **MCP is a standard API to expose sets of Tools that can be hooked to LLMs.**

It is fairly simple to extend an Inference Client – at HF, we have two official client SDKs: `@huggingface/inference` in JS, and `huggingface_hub` in Python – to also act as a MCP client and hook the available tools from MCP servers into the LLM inference. 

But while doing that, came my second realization:

> **Once you have a MCP Client, an Agent is literally just a while loop on top of it.**

In this short post, I will walk you through how I implemented it in JavaScript, how you can adopt MCP too and how it's going to make Agentic AI way simpler going forward.

## How to run the complete demo

Once you have NodeJS (with `pnpm` or `npm`), just run this in a terminal:

```bash
npx @huggingface/mcp-client
```

or if using `pnpm`:

```bash
pnpx @huggingface/mcp-client
```

You'll see my simple agent connect to two distinct MCP servers (running locally), loading their tools, then prompting you for a conversation.

(video)

## The foundation for this: tool calling native suport in LLMs.

What is going to make this whole blogpost very easy is that LLMs have been trained for function calling, aka. tool use.

A tool is defined by its name, a description, and a JSONSchema representation of its parameters.
In some sense, it is an opaque representation of any function's interface, as seen from the outside.

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

The canonical documentation I will link to here is [OpenAI's function calling doc](https://platform.openai.com/docs/guides/function-calling?api-mode=chat).

Inference engines let you pass a list of tools when calling the LLM, and the LLM is free to call zero, one or more of those tools.
As a developer, you run the tools and feed their result back into the LLM to continue the generation.

Note that in the backend (at the inference engine level), the tools are simply passed to the model in a specially-formatted chat_template and then parsed out of the response (using model-specific special tokens) to expose them as tool calls.

## Implementing a MCP client on top of InferenceClient

Now that we know what a tool is in recent LLMs, let us implement the actual MCP client.

The official doc at https://modelcontextprotocol.io/quickstart/client is fairly well-written. You only have to replace any mention of the Anthropic client SDK by any other OpenAI-compatible client SDK. (There is also a llms.txt you can feed into your LLM of choice to help you code along). As a reminder, we use HF's `InferenceClient`.

> The complete code file is [here] if you want to follow along

My `McpClient` class has:
- an Inference Client (works with any Inference Provider, and HfJS supports remote and local endpoints)
- a set of MCP client sessions, one for each connected MCP server
- and a list of available tools that is going to be filled from the connected servers and just lightly formatted.

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

`StdioServerParameters` is an interface from the MCP SDK that will let you easily spawn a local process: currently, all MCP servers are actually local processes (though remote servers are coming soon).

For each MCP server we connect to, we slightly reformat its list of tools and add them to `this.availableTools`.

### How to use the tools

Easy, you just pass `this.availableTools` to your LLM chat-completion (in addition to your typical array of messages):

```ts
const stream = this.client.chatCompletionStream({
	provider: this.provider,
	model: this.model,
	messages,
	tools: this.availableTools,
	tool_choice: "auto",
});
```

When parsing or streaming the output, the LLM will generate some tool calls (a function name, and some JSON-encoded arguments), which you (as a developer) need to compute. The MCP client SDK onces again makes that very easy, it has a `client.callTool()` method:

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

## What is an agent anyway

Now that we have a MCP client capable of connecting to arbitrary MCP servers to get lists of tools and capable of injecting them and parsing them from the LLM inference, well... what is an Agent?

> Once you have an inference client with a set of tools, then a Agent is just a while loop on top of it.


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

By default, we use a very simple system prompt inspired by the one shared in the [GPT-4.1 prompting guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)

Even though this comes from OpenAI, this sentence in particular applies to more and more models – open models too:

> We encourage developers to exclusively use the tools field to pass tools, rather than manually injecting tool descriptions into your prompt and writing a separate parser for tool calls, as some have reported doing in the past.

Then loading the tools is literally just connecting to the MCP servers we want, in parallel:

```ts
async loadTools(): Promise<void> {
	await Promise.all(this.servers.map((s) => this.addMcpServer(s)));
}
```

We add two extra tools that can be used by the LLM for our Agent's control flow:

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

The gist of the Agent's main while loop is that we simply recurse and iterate with the LLM alternating between tool calling and feeding it the tool results, and we do this **until the LLM starts to respond with two non-tool messages in a row**.

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
	// eslint-disable-next-line @typescript-eslint/no-non-null-assertion
	const currentLast = this.messages.at(-1)!;
	debug("current role", currentLast.role);
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




