---
title: "Five Big Improvements to Gradio MCP Servers" 
thumbnail: /blog/assets/gradio-mcp-updates/5_mcp_improvements.png
authors:
- user: freddyaboulton
---

# Five Big Improvements to Gradio MCP Servers

[Gradio](https://gradio.app) is an open-source Python package for creating AI-powered web applications. Gradio is compliant with the [MCP server protocol](https://modelcontextprotocol.io/introduction) and powers thousands of MCP servers hosted on [Hugging Face Spaces](https://hf.co/spaces). The Gradio team is **betting big** on Gradio and Spaces being the best way to build and host AI-powered MCP servers.

To that end, here are some of the big improvements we've added to Gradio MCP servers as of version [5.38.0](https://github.com/gradio-app/gradio/releases/tag/gradio%405.38.0).

## Seamless Local File Support

If you've tried to use a remote Gradio MCP server that takes a file as input (image, video, audio), you've probably encountered this error:

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/MCPError.png">

This happens because the Gradio server is hosted on a different machine, meaning any input files must be accessible via a public URL so they can be downloaded remotely.

While many ways exist to host files online, they all add a manual step to your workflow. In the age of LLM agents, shouldn't we expect them to handle this for you?

Gradio now includes a **"File Upload" MCP server** that agents can use to upload files directly to your Gradio application. If any tools in your Gradio MCP server require file inputs, the connection documentation will now show you how to start the "File Upload" MCP server:

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/MCPConnectionDocs.png">

Learn more about using this server (and important security considerations) in the [Gradio Guides](https://www.gradio.app/guides/file-upload-mcp).

## Real-time Progress Notifications

Depending on the AI task, getting results can take some time. Now, Gradio **streams progress notifications** to your MCP client, allowing you to monitor the status in real-time!

<video src="https://github.com/user-attachments/assets/b507c380-d6b6-4307-b0d1-be423a7414f3" controls></video>

As an MCP developer, it's highly recommended to implement your MCP tools to emit these progress statuses. Our [guide](https://www.gradio.app/guides/building-mcp-server-with-gradio#sending-progress-updates) shows you how.

## Transform OpenAPI Specs to MCP in One Line

If you want to integrate an existing backend API into an LLM, you have to manually map API endpoints to MCP tools. This can be a time consuming and error prone chore. With this release, Gradio can automate the entire process for you! With a single line of code, you can integrate your business backend into any MCP-compatible LLM. 

[OpenAPI](https://www.openapis.org/) is a widely adopted standard for describing RESTful APIs in a machine-readable format, typically as a JSON file. Gradio now features the `gr.load_openapi` function, which creates a Gradio application directly from an OpenAPI schema. You can then launch the app with `mcp_server=True` to automatically create an MCP server for your API!

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

Find more details in the Gradio [Guides](https://www.gradio.app/guides/from-openapi-spec).

## Improvements to Authentication

A common pattern in MCP server development is to use authentication headers to call services on behalf of your users. As an MCP server developer, you want to clearly communicate to your users which credentials they need to provide for proper server usage.

To make this possible, you can now type your MCP server arguments as `gr.Header`. Gradio will automatically extract that header from the incoming request (if it exists) and pass it to your function. The benefit of using `gr.Header` is that the MCP connection docs will automatically display the headers you need to supply when connecting to the server!

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

![MCP Header Connection Page](https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/MCPUploadUpdated.png)

You can read more about this in the Gradio [Guides](https://www.gradio.app/guides/building-mcp-server-with-gradio#using-the-gr-header-class).

## Modifying Tool Descriptions

Gradio automatically generates tool descriptions from your function names and docstrings. Now you can customize the tool description even further with the `api_description` parmeter. In this example, the tool description will read "Apply a sepia filter to any image."

```python
import gradio as gr
import numpy as np

def sepia(input_img):
    """
    Args:
        input_img (np.array): The input image to apply the sepia filter to.

    Returns:
        The sepia filtered image.
    """
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

gr.Interface(sepia, "image", "image", 
             api_description="Apply a sepia filter to any image.")\
            .launch(mcp_server=True)
```

Read more in the [guide](https://www.gradio.app/guides/building-mcp-server-with-gradio#modifying-tool-descriptions).


## Conclusion

Want us to add a new MCP-related feature to Gradio? Let us know in the comments of the blog or on [GitHub](https://github.com/gradio-app/gradio/issues). Also if you've built a cool MCP server or Gradio app let us know in the comments and we'll amplify it!