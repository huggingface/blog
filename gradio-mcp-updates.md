---
title: "Four Ways Gradio MCP Servers Just Got Better" 
thumbnail: /blog/assets/gradio-mcp-updates/<PNG_PATH>.png
authors:
- user: freddyaboulton
---

# Gradio MCP Servers Just Got Better

[Gradio](https://gradio.app) is an open source python package for creating ai-powered web applications. Gradio is compliant with the [MCP server protocol](ADD_LINK) and powers thousands of MCP serves hosted on [Hugging Face Spaces](https://hf.co/spaces). The Gradio team is **betting big** on Gradio and Spaces being the best way to build and host AI-powered MCP servers.

To that end, here are some of the big improvements we've added to Gradio MCP servers as of version [5.38.0](CHECK_VERSION).

## Local File Support

If you've tried to to use a remote Gradio MCP server that takes a file as input (image, video, audio), you've probably run into this error:

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/MCPError.png">

The reason is that since the Gradio server is hosted on a different machine, any input files must be available via a public URL so that they can downloaded in the remote machine.

There are many ways to host files on the internet, but they all require adding a manual step to your workflow. In the age of LLM agents, shouldn't we expect them to handle this step for you?

Gradio now comes with a "File Upload" MCP server that agents can use to upload files to the Gradio application. If any of the tools in the Gradio MCP server require file inputs, the connection docs will show how you can start the "File Upload" MCP server:

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/MCPConnectionDocs.png">

Read more about how to use this server (as well as important security considerations) in the [Gradio Guides](https://www.gradio.app/guides/file-upload-mcp)

## Go From OpenAPI Spec To MCP In One Line

[OpenAPI](https://www.openapis.org/)** is a widely adopted standard for describing RESTful APIs in a machine-readable format, typically as a JSON file. Gradio now comes with the `gr.load_openapi` function, which creates a Gradio application directly from an OpenAPI schema. You can then launch the app with `mcp_server=True` and automatically create an MCP server for your API!

```python
```python
import gradio as gr

demo = gr.load_openapi(
    openapi_spec="https://petstore3.swagger.io/api/v3/openapi.json",
    base_url="https://petstore3.swagger.io/api/v3",
    paths=["/pet.*"],
    methods=["get", "post"],
)

demo.launch(mcp_server=True)
```

Read more about it in the Gradio [Guides](https://www.gradio.app/guides/from-openapi-spec)

## Improvements to Authentication

A common pattern in MCP server development is to use authentication headers to call services on behalf of your users. As an MCP server developer, you want to let your users know which headers they need to provide to their client so they can properly use your server. 

You can now type your MCP server arguments as `gr.Header`. Gradio will automatically extract that header from the incoming request (if it exists) and pass it to your function. The benefit of using `gr.Header` is that the MCP connection docs will automatically display the headers you need to supply when connecting to the server!

In the example below, the `X-API-Token` header is extracted from the incoming request and passed in as the `x_api_token` argument to `make_api_request_on_behalf_of_user`.

```python
import gradio as gr

def make_api_request_on_behalf_of_user(prompt: str, x_api_token: gr.Header):
    """Make a request to everyone's favorite API.
    Args:
        prompt: The prompt to send to the API.
    Returns:
        The response from the API.
    Raises:
        AssertionError: If the API token is not valid.
    """
    return "Hello from the API" if not x_api_token else "Hello from the API with token!"


demo = gr.Interface(
    make_api_request_on_behalf_of_user,
    [
        gr.Textbox(label="Prompt"),
    ],
    gr.Textbox(label="Response"),
)

demo.launch(mcp_server=True)
```

![MCP Header Connection Page](https://github.com/user-attachments/assets/e264eedf-a91a-476b-880d-5be0d5934134)

You can read more about this in the Gradio [Guides](https://www.gradio.app/guides/building-mcp-server-with-gradio).

## Progress Notifications

Depending on the AI task being performed, your results may take some seconds to be ready. Now Gradio will stream progress notifications to your MCP client so you can monitor its progress!

<video src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/ProgressNotifications.mp4" controls>

As an MCP developer, it's good to implement your MCP tools so that they emit progress statuses. Learn how you can do this by reading our [guide](add-link) 

## Conclusion

Want us to add a new MCP-related feature to Gradio? Let us know in the comments of the blog or on [GitHub](https://github.com/gradio-app/gradio/issues)