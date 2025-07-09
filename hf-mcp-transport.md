# Building the Hugging Face MCP Server

Authors: Shaun, Julien, Franck, Abubakar, Elliot.

**tldr;** The Hugging Face Official MCP Server offers unique customization options enabling LLMs to access the Hub as well as 1000s of Gradio applications through a single endpoint. We deployed it as a Remote Server using the Streamable HTTP Transport and examine the options that MCP developers can choose from.

## Introduction

The Model Context Protcol (MCP) is fulfilling its promise of being the standard to connect AI Assistants to the outside world. 

At Hugging Face, providing access to the Hub via MCP is an obvious choice, and this article shares some of our experience developing the `hf.co/mcp` MCP Server.

## Design Choices

The community uses the Hub for a wide range of tasks including Research, Development, Content Creation and more. We wanted to let people customise the Server for their own needs and give people simple access to 1000s of Gradio Applications. This meant making the MCP Server dynamic - adjusting it's Tools for each the specific User on the fly.

We also wanted to make access simple - avoiding complicated downloads and configuration - so making it remotely accessible via a simple URL was a must.

## Remote Servers

When building a remote MCP Server, the first decision is deciding how Clients will connect to it.  MCP offers several transport options, with different trade-offs.

Here is a brief summary of the Transport Options offered by the Model Context Protocol: 

| Transport         | Notes                                                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `STDIO`           | Typically used when the MCP Server is running on the same computer as the Client. Able to access to local resources such as files if needed. |
| `HTTP with SSE`   | Used for remote connections over HTTP. Now deprecated in the 2025-03-26 version of MCP but still in use.                                     |
| `Streamable HTTP` | A more flexible remote HTTP transport that provides more options for deployment than the outgoing SSE version                                |

Both `STDIO` and `HTTP with SSE` are fully bi-directional by default - meaning that Client and Server maintain an open connection and can send messages to each other at any time. 

Since launching in November 2024, MCP has undergone rapid evolution with 3 revisions the protocol in 9 months. This has seen the replacement of the SSE Transport with Streamable HTTP, as well as the introduction and then rework of Authorization.

These rapid changes mean support for different MCP Features and revisions in Client applications varies, providing extra challenges for our design choices.

#### Understanding Streamable HTTP

> [!TIP]
> SSE refers to "Server Sent Events" - a way for HTTP Servers to maintain an open connection and send events in response to a request.

MCP Server Developers face lots of choices when setting up the new Streamable HTTP transport.

With Streamable HTTP there are 3 main communication patterns to choose from:

- **Direct Response** Simple Request/Response (like standard REST APIs). This is perfect for straightforward, stateless operations such as calculations or searches.
- **Request-Scoped Streams** Temporary SSE Streams associated with a single Request. This is useful for [Progress Updates](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/progress) if the Tool Call takes a long time, or the Tool needs to request information from the User with an [Elicitation](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation).
- **Server Push Streams** Long-lived SSE connection supporting server-initiated messages. This enables Resource, Tool and Prompt List change notifications or ad-hoc Sampling. These connections need some extra management such as [keep-alive](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/ping) and resumption mechanics on re-connection. 

> [!TIP]
> When using Request-Scoped Streams, use the `sendNotification()` and `sendRequest()` methods provided in the `RequestHandlerExtra` parameter (TypeScript SDK) or set `related_request_id` (Python SDK) to ensure messages are sent on the correct stream.

An additional factor to consider is whether or not the MCP Server needs to maintain state for each connection. This is decided by the Server when the Client sends its Initialize request:

| | Stateless | Stateful |
|---|-----------|----------|
| **Session IDs** | Not needed | Required (`mcp-session-id` header) |
| **What it means** | Each request is independent | Server maintains client context |
| **Scaling** | Simple horizontal scaling: any instance can handle any request | Need session affinity or shared state mechanisms |


> The [Hugging Face MCP Server](https://github.com/evalstate/hf-mcp-server) supports STDIO, SSE and Streamable HTTP deployment in all of these deployment scenarios. You can configure Ping keep-alive and last activity timeouts for Server Push Streams. There is also a built in observability dashboard you can use to understand how different Clients manage connections. It also allows you to select tools and get Tool Change Notifications.

> SCREENSHOT Show connection dashboard here

The below table summarises which MCP Features require which communications pattern:

| MCP Feature             | Server Push                  | Request Scoped                         | Direct Response |
| -------------------------- | ---------------------------- | -------------------------------------- | -------- |
| Tools, Prompts, Resources | Y                            | Y                                      | Y |
| Sampling/Elicitations      | Server Initiated at any time | Related to a  Client initated requests | N        |
| Resource Subscriptions     | Y                            | N                                      | N        |
| Tool/Prompt List Changes   | Y                            | N                                      | N        |
| Tool Progress Notification | -                            | Y                                      | N        |


### Our Deployment

For production, we decided to launch our MCP Server with Streamable HTTP in a stateless, Direct Response configuration for the following reasons:

**Stateless** Our User state is the currently selected tools and Gradio endpoints, as well as the the User ZeroGPU account quota. This is all managed using the supplied `HF_TOKEN` or OAuth credentials - or with a standard set for Anonymous Users. At the moment none of our MCP Server tools require us to maintain Client state between requests.

> [!TIP]
> You can use OAuth login by adding `?login` to the MCP Server url - e.g. `https://huggingface.co/mcp?login`. This may become the default once `claude.ai` remote integration supports the latest OAuth spec.

**Direct Response** This provides the lowest deployment overhead, and at the moment we don't have any Tools that require Sampling or Elicitation during execution.

**Future Support** At launch, SSE was still the default transport in a lot of MCP Clients - however we didn't want to invest heavily in managing it due to it's imminent deprecation. Fortunately popular clients had already made the switch (VSCode and Cursor), and within a week of launch `claude.ai` also added support. If you need to connect with SSE, feel free to deploy a copy of our Server on a [FreeCPU Hugging Face Space](https://huggingface.co/new-space).

One thing we would love is to support are real-time Tool List Changed notifications when people update their settings on the Hub - however this raises a practical issue. 

Users tend to configure their favourite MCP Servers in their Client and leave them enabled. This means that the Client will connect whilst the application is open. Well behaved clients may shut the connection periodically, resuming when necessary - and potentially receiving queued updates. In practice, it is far simpler to simply refresh the connection than maintain complex state and keep-alive mechanisms for this scenario. Additionally, if the connection were succesfully kept alive this would mean maintaining connections for every User that has their Client open and the Hugging Face Server enabled.

Unless you have reasonable control over the Client/Server pair, using **Server Push Streams** adds a lot of complexity to a public deployment, when simpler solutions for refreshing the Tool List exist.

### In Production Reflections

#### URL User Experience

Just before launch, @Julien-C asked if it would be possible to include a page to provide friendly instructions for users that visit the URL. This improves the overall User Experience a lot - the default response returns an unfriendly bit of JSON. 

Initially, we found this generated an enormous amount of traffic. After a bit of investigation we found that when returning a web page rather than an error, VSCode would poll the endpoint multiple times per second! 

The fix suggested by @ElliotC was to properly detect browsers and only return the page in that circumstance. Fortunately the VSCode team have also rapdily updated  to improve the client behaviour. Although not specifically mentioned, this seems allowable within the MCP Specification. 

#### MCP Client Behaviour

The MCP Protocol sends quite a lot of traffic during initialization. A typical sequence when connecting to an MCP Server would be: `Initialize`, `Notifications/Initialize`, `tools/list` and `prompts/list`. 

Bearing in mind that Clients may reconnect, and Users may complete their session without making a Tool Request. We often find the ratio to be around 100 MCP Control messages for each Tool Call.

Some clients also send requests that don't make sense for our Stateless, Direct Response configuration - for example Pings, Cancellations and attempts to list Resources.

The first week of July 2025 saw an astonishing 164 different Clients accessing our Server. Interestingly, one of the most popular tools is [`mcp-remote`](https://github.com/geelen/mcp-remote). Approximately half of all Clients use it as a bridge to connect to our remote server. 

## Conclusion

MCP is rapidly evolving, and we're excited about what has already been achieved across Chat Applications, IDEs, Agents and MCP Servers over the last few months.

We can already see how powerful integrating the Hugging Face Hub has been (ADD LINKS TO DEMOS ETC.). Support for Gradio Spaces now makes it possible for LLMs to be extended with the latest Machine Learning applications. 

We hope that this post has provided insights to the decisions that need to be made building Remote MCP Servers, and encourage you to try some of the examples in your favourite MCP Client (MORE DEMO LINKS - MAYBE A LIST AT THE END). Also, take a look at our [Open Source MCP Server](https://github.com/evalstate/hf-mcp-server), try some of the different transport options with your Client or open an Issue or Pull Request to make improvements or suggest new functionality.  

