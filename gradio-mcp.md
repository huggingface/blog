---
title: "How to Build an MCP Server with Gradio!"
thumbnail: /blog/assets/gradio-1m/thumbnail.png
authors:
- user: abidlabs
---

# How to Build an MCP Server in 5 Lines of Python

Large language models (LLMs) are  with external tools and data sources is becoming increasingly essential. This guide will show you how to leverage Gradio—a Python library trusted by over 1 million developers worldwide—to build a powerful Model Context Protocol (MCP) server with minimal code.

Gradio has become one of the most popular open-source projects in the AI ecosystem, powering applications like Automatic1111, Dall-E Mini, and LLaMA-Factory. Its success stems from providing intuitive building blocks that make AI accessible to developers of all skill levels.

Punchline: it's as simple as setting `mcp_server=True` in `.launch()`. Let's dive in!

### Prerequisites

If not already installed, please install Gradio with the MCP extra:

```bash
pip install gradio[mcp]
```

This will install the necessary dependencies, including the `mcp` package. You'll also need a LLM application that supports tool calling using the MCP protocol, such as Claude Desktop, Cursor, or Cline (these are known as "MCP Clients").

## What is an MCP Server?

The Model Context Protocol (MCP) addresses a fundamental challenge in AI by providing a standardized way for LLMs to connect with external data sources and tools—essentially a "universal remote" for AI applications. Released by Anthropic as an open-source protocol, MCP builds on existing function calling capabilities but eliminates the need for custom integration between each LLM and external system.

Prior to MCP, connecting AI models to external tools was a complex, fragmented process. Each API endpoint had to be individually documented and integrated, forcing models to memorize specific details about each tool's parameters, authentication requirements, and error handling.

MCP extends traditional LLM function calling by separating function definitions from LLM applications. This allows tool builders to create standardized interfaces (MCP servers) for their tools and services, while LLM agents can leverage existing MCP servers instead of reimplementing them from scratch.

Gradio—which began as a simple tool to demo machine learning models—has evolved into a comprehensive framework for building and sharing AI applications. With over 6.7 million downloads per month, it has become a staple in the machine learning community, allowing developers to go beyond code and present their work in an engaging and tangible way.

## Example: Counting Letters in a Word

LLMs are famously not great at counting the number of letters in a word (e.g., the number of "r"s in "strawberry"). But what if we equip them with a tool to help? Let's start by writing a simple Gradio app that counts the number of letters in a word or phrase:

```python
import gradio as gr

def letter_counter(word, letter):
    """Count the occurrences of a specific letter in a word.
    
    Args:
        word: The word or phrase to analyze
        letter: The letter to count occurrences of
        
    Returns:
        The number of times the letter appears in the word
    """
    return word.lower().count(letter.lower())

demo = gr.Interface(
    fn=letter_counter,
    inputs=["text", "text"],
    outputs="number",
    title="Letter Counter",
    description="Count how many times a letter appears in a word"
)

demo.launch(mcp_server=True)
```

Notice that we have set `mcp_server=True` in `.launch()`. This is all that's needed for your Gradio app to serve as an MCP server! Now, when you run this app, it will:

1. Start the regular Gradio web interface
2. Start the MCP server
3. Print the MCP server URL in the console

This showcases Gradio's core strength: allowing you to create customizable web interfaces for your machine learning models with Python. Whether you have your own model or want to try out a new open-source LLM without any hassle, Gradio provides an intuitive solution for both new and experienced developers.

The MCP server will be accessible at:
```
http://your-server:port/gradio_api/mcp/sse
```

Gradio automatically converts the `letter_counter` function into an MCP tool that can be used by LLMs. The docstring of the function will be used to generate the description of the tool and its parameters.

This integration showcases how Function Calling and MCP work together: Function Calling translates prompts into structured instructions, while MCP handles the execution of those instructions, ensuring seamless AI integration with external tools.

All you need to do is add this URL endpoint to your MCP Client (e.g., Claude Desktop, Cursor, or Cline), which typically means pasting this config in the settings:

```
{
  "mcpServers": {
    "gradio": {
      "url": "http://your-server:port/gradio_api/mcp/sse"
    }
  }
}
```

(By the way, you can find the exact config to copy-paste by going to the "View API" link in the footer of your Gradio app, and then clicking on "MCP").

## Key features of the Gradio <> MCP Integration

1. **Tool Conversion**: Each API endpoint in your Gradio app is automatically converted into an MCP tool with a corresponding name, description, and input schema. To view the tools and schemas, visit http://your-server:port/gradio_api/mcp/schema or go to the "View API" link in the footer of your Gradio app, and then click on "MCP".

   Gradio's ability to handle dynamic UI manipulation allows developers to create sophisticated and responsive interfaces using simple Python code. This is especially valuable when building tools that need to provide immediate visual feedback.

2. **Environment variable support**. There are two ways to enable the MCP server functionality:

   *  Using the `mcp_server` parameter, as shown above:
      ```python
      demo.launch(mcp_server=True)
      ```

   * Using environment variables:
      ```bash
      export GRADIO_MCP_SERVER=True
      ```

3. **File Handling**: The server automatically handles file data conversions, including:
   - Converting base64-encoded strings to file data
   - Processing image files and returning them in the correct format
   - Managing temporary file storage

    Recent updates to Gradio have significantly improved its image handling capabilities, adding features like Photoshop-style zoom and pan, full transparency control, and custom layers—injecting new vitality into the image processing capabilities of AI applications.

    It is **strongly** recommended that input images and files be passed as full URLs ("http://..." or "https:/...") as MCP Clients do not always handle local files correctly.

4. **Hosted MCP Servers on 󠀠🤗 Spaces**: You can publish your Gradio application for free on Hugging Face Spaces, which will allow you to have a free hosted MCP server. Gradio is part of a broader ecosystem that includes Python and JavaScript libraries for building machine learning applications or querying them programmatically, making it much more than just a UI framework.

Here's an example of such a Space: https://huggingface.co/spaces/abidlabs/mcp-tools. Notice that you can add this config to your MCP Client to start using the tools from this Space immediately:

```
{
  "mcpServers": {
    "gradio": {
      "url": "https://abidlabs-mcp-tools.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

## Beyond UI: Gradio as an AI Integration Platform

Gradio's framework offers more than just a pretty interface—it provides a complete solution for rapidly building proof-of-concept AI applications. The integration with MCP enables organizations to deploy engaging, low-effort generative AI experiences that can impress customers and inspire development teams.

For Python developers who need quick results without worrying about UI aesthetics, Gradio provides an efficient way to produce a web interface with minimal code. The components stack sensibly in a responsive manner, making it ideal for showcasing machine learning models.

By combining Gradio's intuitive interface-building capabilities with MCP's standardized tool protocol, you're creating a powerful bridge between LLMs and your custom functionality. This opens up countless possibilities for AI-enhanced applications across various domains.

<video src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/mcp_guide1.mp4" style="width:100%" controls preload> </video>