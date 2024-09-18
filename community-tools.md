---
title: "Introducing Community Tools on HuggingChat"
thumbnail: /blog/assets/community-tools/thumbnail.png
authors:
  - user: nsarrazin
---

<div align="center">
    <video alt="demo.mp4" controls autoplay loop autobuffer muted playsinline>
        <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/demo.webm" type="video/webm">
    </video>
</div>

Today we’re releasing our latest feature on HuggingChat: Community Tools! This lets you turn any Space that you love on HuggingFace into a tool that can be used by models directly from HuggingChat.

With this feature, we’re also expanding the modalities available in HuggingChat. You can now use community tools to understand images, generate videos, or answer with a text-to-speech model. The possibilities are endless and anyone can create tools using Spaces on Hugging Face! Explore existing tools [here](https://huggingface.co/chat/tools).

In this post we’re going to look at a few use cases for creating community tools:

1. [Turning a community Space into a tool](#turning-a-community-Space-into-a-tool)
2. [Creating a custom tool yourself](#creating-a-custom-tool-yourself)
3. [Enhance your assistants with community tools](#enhance-your-assistants-with-community-tools)
4. [Create a RAG tool on your own documents](#create-a-rag-tool-on-your-own-documents)

## Turning a community Space into a tool

You can turn anyone’s public Space into a tool. This is handy for using the latest models directly in HuggingChat. Let’s use [DamarJati/FLUX.1-RealismLora](https://huggingface.co/spaces/DamarJati/FLUX.1-RealismLora) as an example here.

Start by [creating a new tool](https://huggingface.co/chat/tools/new) and filling in the fields. As soon as you input the Space URL into the _Hugging Face Space URL_ field you’ll see the available functions and parameters get filled automatically.

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/tools-step-1.png"/>
</div>

There are some fields that need to be filled in correctly in order to ensure optimal tool performance.

- **Tool Description:** The description is passed to the LLM to explain what the tool can do. Keep it short and describe what the tool can be used for.
- **AI Function Name:** The tools are represented as code functions. This is your tool’s function name. Keep it short, unique and self-explanatory.
- **Arguments:** These are the tool parameters the LLM can fill-in. They can be:
  - **Required:** The model must fill in a value to use this tool. This required the parameter to be described properly.
  - **Optional:** There is a default value provided but the model can override it if needed.
  - **Fixed:** The value is fixed when creating the tool and can’t be changed by the model.

You can always look at other tools’ definition to better understand how to create tools. [(example)](https://huggingface.co/chat/tools/000000000000000000000001/edit)

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/tools-step-2.png"/>
</div>

Now that our tool is created, we can enable it and start using it with tools compatible models!

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/tools-step-3.png"/>
</div>

## **Creating a custom tool yourself**

Using existing Spaces can cover a lot of use cases but if you can write basic python then you can just as easily create custom tools for yourself. Let’s do a simple example of a dice roll tool since LLMs are quite bad at picking random numbers by themselves.

Start by creating a [new Gradio Space](https://huggingface.co/new-space?sdk=gradio) on Hugging Face. The _CPU Basic_ free tier works fine for this. Your Space will have to be public in order to be used inside HuggingChat.

Create a simple [`app.py`](http://app.py) app inside your pace repo, in the case of our roll dice example that could be:

```python
import gradio as gr
import random

def roll_dice(sides=6):
    return random.randint(1, sides)

demo = gr.Interface(
    fn=roll_dice,
    inputs=gr.Number(value=6, label="Number of Sides"),
    outputs="text",
    title="Dice Roller",
    description="Enter the number of sides for the dice and get the roll result."
)

demo.launch()
```

If you’re not familiar with Gradio, it’s very easy to get started creating interfaces, you can find the docs [here](https://www.gradio.app/docs/gradio/interface).

> [!TIP]
> You can have multiple functions inside a single Space to make it easier to manage your tools.

Once you’re done, push the changes and when your Space is deployed you can create a community tool for it in HuggingChat just like we did previously.

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/tools-custom.png"/>
</div>

## Enhance your assistants with Community Tools

You can activate tools directly by going to [the tools page](https://huggingface.co/chat/tools) and picking the ones you like or you can also package tools in an assistant.

When [creating an assistant](https://huggingface.co/chat/settings/assistants/new) using a model that is compatible with tool calling you will now have the option to select tools. Search for the name of the tools and add up to 3 different tools. In our case, let’s create a Game Master assistant that has access to image generation and dice rolling tools.

> [!TIP]
> You can use the system instructions field to tell the model when to use the tools.

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/tools-assistant.png"/>
</div>

## Create a RAG tool on your own documents

To go along with the release we created a simple RAG tool that you can easily copy to ask questions about your documents directly from HuggingChat. Start by duplicating [this Space](https://huggingface.co/spaces/nsarrazin/rag-tool-template) into your own account. You can then drop files you want to be parsed in the `sources/` folder of that Space.

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/tools-rag-1.png"/>
</div>

Once the Space is started up you can easily create a tool out of it on HuggingChat like we’ve covered previously.

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/community-tools/tools-rag-2.png"/>
</div>

### Share your feedback with us

With the release of Community Tools we hope you’ll be able to enhance your chats with multimodal content and custom tools. The feature is still experimental so if you see Spaces that are not supported or tools that don’t work, please share them with us in [the feedback thread](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/569)!
