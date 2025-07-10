---
title: "Building the Hugging Face MCP Server" 
thumbnail: /blog/assets/building-hf-mcp/building-hf-mcp.png
authors:
- user: evalstate
- user: julien-c
- user: coyotte508
- user: abidlabs
---

# Building the Hugging Face MCP Server

> [!TIP]
> **TL;DR:** The Hugging Face Official MCP Server offers unique customization options for AI Assistants accessing the Hub, along with access to thousands of AI applications through one simple URL. We used MCPs "Streamable HTTP" transport for deployment, and examine in detail the trade-offs that Server Developers have. 
>
> We've learned many things about building a useful MCP server in the last month - we'll describe our journey here.

## Introduction

The Model Context Protocol (MCP) is fulfilling its promise of being the standard to connect AI Assistants to the outside world. 

At Hugging Face, providing access to the Hub via MCP is an obvious choice, and this article shares our experience developing the [`hf.co/mcp`](https://hf.co/mcp) MCP Server.

## Design Choices

The community uses the Hub for research, development, content creation and more. We wanted to let people customize the server for their own needs, as well as easily access thousands of AI applications available on Spaces. This meant making the MCP Server dynamic by adjusting users' tools on the fly.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hf-mcp-remote/hf-mcp-settings.png" alt="The Hugging Face MCP Settings Page">
  <figcaption>The <a href="https://huggingface.co/settings/mcp">Hugging Face MCP Settings Page</a> where Users can configure their tools.</figcaption>
</figure>

We also wanted to make access simple by avoiding complicated downloads and configuration, so making it remotely accessible via a simple URL was a must.

## Remote Servers

When building a remote MCP Server, the first decision is deciding how clients will connect to it. MCP offers several transport options, with different trade-offs. **TL;DR:** our open source code supports all variants, but for production we chose to go with the most modern one. This section goes through the different options in detail. 

Since its launch in November 2024, MCP has undergone rapid evolution with 3 protocol revisions in 9 months. This has seen the replacement of the SSE Transport with Streamable HTTP, as well as the introduction and rework of authorization.

These rapid changes mean support for different MCP Features and revisions in Client applications varies, providing extra challenges for our design choices.

Here is a brief summary of the Transport Options offered by the Model Context Protocol and associated SDKs: 

| Transport         | Notes                                                                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `STDIO`           | Typically used when the MCP Server is running on the same computer as the Client. Able to access local resources such as files if needed. |
| `HTTP with SSE`   | Used for remote connections over HTTP. Deprecated in the 2025-03-26 version of MCP but still in use.                                     |
| `Streamable HTTP` | A more flexible remote HTTP transport that provides more options for deployment than the outgoing SSE version                                |

Both `STDIO` and `HTTP with SSE` are fully bi-directional by default - meaning that Client and Server maintain an open connection and can send messages to each other at any time. 

> [!TIP]
> SSE refers to "Server Sent Events" - a way for HTTP Servers to maintain an open connection and send events in response to a request.

#### Understanding Streamable HTTP

MCP Server Developers face a lot of choices when setting up the Streamable HTTP transport.

There are 3 main communication patterns to choose from:

- **Direct Response** - Simple Request/Response (like standard REST APIs). This is perfect for straightforward, stateless operations like simple searches.
- **Request Scoped Streams** - Temporary SSE Streams associated with a single Request. This is useful for sending [Progress Updates](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/progress) if the Tool Call takes a long time - such as Video Generation. Additionally the Server may need to request information from the user with an [Elicitation](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation), or conduct a Sampling request.
- **Server Push Streams** - Long-lived SSE connection supporting server-initiated messages. This enables Resource, Tool and Prompt List change notifications or ad-hoc Sampling and Elicitations. These connections need extra management such as [keep-alive](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/ping) and resumption mechanics on re-connection. 

> [!TIP]
> When using Request Scoped Streams with the official SDKs, use the `sendNotification()` and `sendRequest()` methods provided in the `RequestHandlerExtra` parameter (TypeScript) or set the `related_request_id` (Python) to send messages to the correct stream.

An additional factor to consider is whether or not the MCP Server itself needs to maintain state for each connection. This is decided by the Server when the Client sends its Initialize request:

| | Stateless | Stateful |
|---|-----------|----------|
| **Session IDs** | Not needed | Server responds with an `mcp-session-id` |
| **What it means** | Each request is independent | Server maintains client context |
| **Scaling** | Simple horizontal scaling: any instance can handle any request | Need session affinity or shared state mechanisms |
| **Resumption** | Not needed | May replay messages for broken connections |

The table below summarizes the MCP Features and their supported communication pattern:

| MCP Feature             | Server Push                  | Request Scoped                         | Direct Response |
| -------------------------- | ---------------------------- | -------------------------------------- | -------- |
| Tools, Prompts, Resources | Y                            | Y                                      | Y |
| Sampling/Elicitation      | Server Initiated at any time | Related to a Client initiated request | N        |
| Resource Subscriptions     | Y                            | N                                      | N        |
| Tool/Prompt List Changes   | Y                            | N                                      | N        |
| Tool Progress Notification | -                            | Y                                      | N        |

With Request Scoped streams, Sampling and Elicitation requests need a Stateful connection so that the `mcp-session-id` can be used for response association.

> [!TIP]
> The [Hugging Face MCP Server](https://github.com/evalstate/hf-mcp-server) is Open Source - and supports STDIO, SSE and Streamable HTTP deployment in both Direct Response and Server Push mode. You can configure keep-alive and last activity timeouts when using Server Push Streams. There's also a built-in observability dashboard that you can use to understand how different Clients manage connections, and handle Tool List change notifications.

The following picture shows our MCP Server connection dashboard running in "Server Push" Streamable HTTP mode:

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hf-mcp-remote/hf-mcp-connections.png" alt="The Hugging Face MCP Server Connection Dashboard">
  <figcaption>The Hugging Face MCP Server Connection Dashboard.</figcaption>
</figure>


### Production Deployment

For production, we decided to launch our MCP Server with Streamable HTTP in a Stateless, Direct Response configuration for the following reasons:

**Stateless** For anonymous users we supply a standard set of Tools for using the Hub along with an Image Generator. For authenticated users our state comprises their [selected tools](https://huggingface.co/settings/mcp) and chosen Gradio applications. We also make sure that users ZeroGPU quota is correctly applied for their account. This is managed using the supplied `HF_TOKEN` or OAuth credentials that we  look up on request. None of our existing tools require us to maintain any other state between requests.

> [!TIP]
> You can use OAuth login by adding `?login` to the MCP Server url - e.g. `https://huggingface.co/mcp?login`. We may make this the default once the `claude.ai` remote integration supports the latest OAuth spec.

**Direct Response**  provides the lowest deployment resource overhead - and we don't currently have any Tools that require Sampling or Elicitation during execution.

**Future Support** At launch, the "HTTP with SSE" transport was still the remote default in a lot of MCP Clients. However, we didn't want to invest heavily in managing it due to its imminent deprecation. Fortunately, popular clients had already started making the switch (VSCode and Cursor), and within a week of launch `claude.ai` also added support. If you need to connect with SSE, feel free to deploy a copy of our Server on a [FreeCPU Hugging Face Space](https://huggingface.co/new-space).

#### Tool List Change Notifications

In the future, we would like to support real-time Tool List Changed notifications when users update their settings on the Hub. However, this raises a couple of practical issues:

First, users tend to configure their favourite MCP Servers in their Client and leave them enabled. This means that the Client remains connected whilst the application is open. Sending notifications would mean maintaining as many open connections as there were currently active Clients - regardless of active usage - on the chance the user updates their tool configuration. 

Second, most MCP Servers and Clients disconnect after a period of inactivity, resuming when necessary. This inevitably means that immediate push notifications would be missed - as the notification channel will have been closed. In practice, it is far simpler for the Client to refresh the connection and Tool List as needed.

Unless you have reasonable control over the Client/Server pair, using **Server Push Streams** adds a lot of complexity to a public deployment, when lower-resource solutions for refreshing the Tool List exist.

#### URL User Experience

Just before launch, [`@julien-c`](https://huggingface.co/julien-c) submitted a PR to include friendly instructions for users visiting `hf.co/mcp`. This hugely improves the User Experience - the default response is otherwise an unfriendly bit of JSON.

Initially, we found this generated an enormous amount of traffic. After a bit of investigation we found that when returning a web page rather than an HTTP 405 error, VSCode would poll the endpoint multiple times per second! 

The fix suggested by [`@coyotte508`](https://huggingface.co/coyotte508) was to properly detect browsers and only return the page in that circumstance. Thanks also to the VSCode team who rapidly [fixed it](https://github.com/microsoft/vscode/pull/251288/files). 

Although not specifically stated - returning a page in this manner _does_ seem acceptable within the MCP Specification. 

#### MCP Client Behaviour

The MCP Protocol sends several requests during initialization. A typical connection sequence is: `Initialize`, `Notifications/Initialize`, `tools/list` and then `prompts/list`. 

Given that MCP Clients will connect and reconnect whilst open, and the fact that users make periodic calls, we find there is a ratio of around 100 MCP Control messages for each Tool Call.

Some clients also send requests that don't make sense for our Stateless, Direct Response configuration - for example Pings, Cancellations or attempts to list Resources (which isn't a capability we currently advertise).

The first week of July 2025 saw an astonishing 164 different Clients accessing our Server. Interestingly, one of the most popular tools is [`mcp-remote`](https://github.com/geelen/mcp-remote). Approximately half of all Clients use it as a bridge to connect to our remote server. 

## Conclusion

MCP is rapidly evolving, and we're excited about what has already been achieved across Chat Applications, IDEs, Agents and MCP Servers over the last few months.

We can already see how powerful integrating the Hugging Face Hub has been and support for Gradio Spaces now makes it possible for LLMs to be easily extended with the latest [Machine Learning applications](https://huggingface.co/blog/gradio-mcp-servers).

Here are some great examples of things people have been doing with our MCP Server so far:
- [Orchestrating Video Production](https://x.com/victormustar/status/1937095435316822244)
- [Image Editing](https://x.com/reach_vb/status/1942247029515735263)
- [Document Searching](https://x.com/NielsRogge/status/1940472422790242561)
- [AI Application Development](https://x.com/llmindsetuk/status/1940358288220336514)
- [Adding Reasoning to existing Models](https://www.linkedin.com/posts/ben-burtenshaw_im-a-big-fan-of-local-models-in-lmstudio-activity-7344001099533590528-zNyw)

We hope that this post has provided insights to the decisions that need to be made building Remote MCP Servers, and encourage you to try some of the examples in your favourite MCP Client.

Take a look at our [Open Source MCP Server](https://github.com/evalstate/hf-mcp-server), and try some of the different transport options with your Client, or open an Issue or Pull Request to make improvements or suggest new functionality.  

Let us know your thoughts, feedback and questions on this [discussion thread](https://huggingface.co/spaces/huggingface/README/discussions/26).