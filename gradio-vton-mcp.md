---
title: "Implementing MCP Servers in Python: An AI Shopping Assistant with Gradio" 
thumbnail: /blog/assets/gradio-vton-mcp/AiAssistantTitle.png
authors:
- user: freddyaboulton
---

# Implementing MCP Servers in Python: An AI Shopping Assistant with Gradio

Python Developers, want to give your LLM superpowers? Gradio is the fastest way to do it! With Gradio's Model Context Protocol (MCP) integration, your LLM can plug directly into the thousands of AI models and Spaces hosted on the Hugging Face [Hub](https://hf.co). By pairing the general reasoning capabilities of LLMs with the specialized abilities of models found on Hugging Face, your LLM can go beyond simply answering text questions to actually solving problems in your daily life.

For Python developers, Gradio makes implementing powerful MCP servers a breeze, offering features like:
* **Automatic conversion of python functions into LLM tools:** Each API endpoint in your Gradio app is automatically converted into an MCP tool with a corresponding name, description, and input schema. The docstring of your function is used to generate the description of the tool and its parameters.
* **Real-time progress notifications:** Gradio streams progress notifications to your MCP client, allowing you to monitor the status in real-time without having to implement this feature yourself.
* **Automatic file uploads**, including support for public URLs and handling of various file types.

Imagine this: you hate shopping because it takes too much time, and you dread trying on clothes yourself. What if an LLM could handle this for you? In this post, we'll create an LLM-powered AI assistant that can browse online clothing stores, find specific garments, and then use a virtual try-on model to show you how those clothes would look on you. See the demo below:

<video src="https://github.com/user-attachments/assets/e5bc58b9-ca97-418f-b78b-ce38d4bb527e" controls alt="AI Shopping Assistant Demo using Gradio python sdk and MCP"></video>

## The Goal: Your Personal AI Stylist

To bring our AI shopping assistant to life, we'll combine three key components:

1. [IDM-VTON](https://huggingface.co/yisol/IDM-VTON) Diffusion Model: This AI model is responsible for the virtual try-on functionality. It can edit existing photos to make it appear as if a person is wearing a different garment. We'll be using the Hugging Face Space for IDM-VTON, accessible [here](https://huggingface.co/spaces/yisol/IDM-VTON).

2. Gradio: Gradio is an open-source Python library that makes it easy to build AI-powered web applications and, crucially for our project, to create MCP servers. Gradio will act as the bridge, allowing our LLM to call the IDM-VTON model and other tools.

3. Visual Studio Code's AI Chat Feature: We'll use VS Code's built-in AI chat, which supports adding arbitrary MCP servers, to interact with our AI shopping assistant. This will provide a user-friendly interface for issuing commands and viewing the virtual try-on results.

## Building the Gradio MCP Server
The core of our AI shopping assistant is the Gradio MCP server. This server will expose one main tool:

1. `vton_generation`: This function will take a human model image and a garment image as input and use the IDM-VTON model to generate a new image of the person wearing the garment.


Here's the Python code for our Gradio MCP server:

```python
from gradio_client import Client, handle_file
import gradio as gr
import re


client = Client("freddyaboulton/IDM-VTON",
                hf_token="<Your-token>")


def vton_generation(human_model_img: str, garment: str):
    """Use the IDM-VTON model to generate a new image of a person wearing a garment."""
    """
    Args:
        human_model_img: The human model that is modelling the garment.
        garment: The garment to wear.
    """
    output = client.predict(
        dict={"background": handle_file(human_model_img), "layers":[], "composite":None},
        garm_img=handle_file(garment),
        garment_des="",
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )

    return output[0]

vton_mcp = gr.Interface(
    vton_generation,
    inputs=[
        gr.Image(type="filepath", label="Human Model Image URL"),
        gr.Image(type="filepath", label="Garment Image URL or File")
    ],
    outputs=gr.Image(type="filepath", label="Generated Image")
)

if __name__ == "__main__":
    vton_mcp.launch(mcp_server=True)
```

By setting mcp_server=True in the `launch()` method, Gradio automatically converts our Python functions into MCP tools that LLMs can understand and use. The docstrings of our functions are used to generate descriptions of the tools and their parameters.

> [!TIP]
> The original IDM-VTON space was implemented with Gradio 4.x which precedes the automatic MCP functionality. So in this demo, we'll be building a Gradio interface that queries the original space via the Gradio API client.

Finally, run this script with python.

## Configuring VS Code

To connect our Gradio MCP server to VS Code's AI chat, we'll need to edit the `mcp.json` file. This configuration tells the AI chat where to find our MCP server and how to interact with it.

You can find this file by typing `MCP` in the command panel and selecting `MCP: Open User Configuration`. Once you open it, make sure the following servers are present:

```json
{
  "servers": {
  "vton": {
    "url": "http://127.0.0.1:7860/gradio_api/mcp/"
  },
  "playwright": {
    "command": "npx",
    "args": [
      "-y",
      "@playwright/mcp@latest"
    ]
   }
  }
}
```

The playwright MCP server will let our AI assistant browse the web.

> [!TIP]
> Make sure the URL of the `vton` server matches the url printed to the console in the previous section. To run the playwright MCP server, you need to have node [installed](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm).

## Putting It All Together

Now we can start interacting with our AI shopping assistant. Open a new chat in VS Code, and you can ask the assistant something like "Browse the Uniqlo website for blue t-shirts, and show me what I would look like in three of them, using my photo at [your-image-url]."

See the above video for an example!

## Conclusion

The combination of Gradio, MCP, and powerful AI models like IDM-VTON opens up exciting possibilities for creating intelligent and helpful AI assistants. By following the steps outlined in this blog post, you can build your own assistant to solve the problems you care most about!