---
title: "Upskill your LLMs With Gradio MCP Servers" 
thumbnail: /blog/assets/upskill-llms-with-gradio-mcp/UpskillThumbnail.png
authors:
- user: freddyaboulton
---

# Upskill your LLMs with Gradio MCP Servers

Upskill your LLMs With Gradio MCP Servers
Have you ever wanted your favorite Large Language Model (LLM) to do more than just answer questions? What if it could edit images for you, browse the web, or organize your email inbox?

Well, now it can! In this blog post, I'll show you:

- What the MCP protocol is and how it works similarly to the smartphone apps we're all used to, but for LLMs.
- How you can find thousands of MCP servers via the "MCP App Store."
- How to add one of these servers to your favorite LLM of choice to grant it a new ability. We'll work through an example using [Flux.1 Kontext[dev]](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Kontext-Dev) which edits images from plain text instructions.


## A Brief Intro To MCP

The **Model Context Protocol (MCP)** is an open standard that enables developers to build secure, two-way connections between an LLM and a set of tools. For example, if you create an MCP server that exposes a tool capable of transcribing a video, then you can connect an LLM client (such as Cursor, Claude Code, or Cline) to the server. The LLM will then know how to transcribe videos and use this tool for you depending on your request.

In short, an MCP server is a standard way to upskill your LLM by granting it a new ability. Think of it like the apps on your smartphone. On its own, your smartphone can't edit images, but you can download an app to do this from the app store. Now, if only there were an app store for MCP servers? 🤔

## Hugging Face Spaces: The MCP App Store

Hugging Face [Spaces](https://hf.co/spaces) is the world's largest collection of AI applications. Most of these spaces perform a specialized task with an AI model. For example:

- [Image Background Removal](https://huggingface.co/spaces/not-lain/background-removal)
- [OCR](https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR)
- [Text-to-Speech Synthesis](https://huggingface.co/spaces/ResembleAI/Chatterbox)

These spaces are implemented with [Gradio](https://gradio.app), an open source python package for creating AI-powered web servers. As of version `5.28.0`, **Gradio apps support the MCP protocol.**

That means that Hugging Face Spaces is the one place where you can find thousands of AI-powered abilities for your LLM, aka the **MCP App Store!**

Want to browse the app store? Visit this [link](https://huggingface.co/spaces?filter=mcp-server). Manually, you can filter for `MCP Compatible` in `https://hf.co/spaces`.

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/MCPFilter.png">

## An Example: An LLM that can edit images

[Flux.1 Kontext[dev]](https://huggingface.co/spaces/black-forest-labs/FLUX.1-Kontext-Dev) is an impressive model that can edit an image from a plain text prompt. For example, if you ask it to "dye my hair blue" and upload a photo of yourself, the model will return the photo but with you having blue hair!

Let's plug this model as an MCP server into an LLM and have it edit images for us. Follow these steps:

1. Go to [Hugging Face](https://huggingface.co/welcome) and create a free account.
2. In your [settings](https://huggingface.co/settings/profile), on the left hand side click on `MCP`. You may have to scroll down in the page to see it.
3. Now, scroll to the bottom of the page. You should see a section called `Spaces Tools`. In the search bar, type `Flux.1-Kontext-Dev` and select the space called `black-forest-labs/Flux.1-Kontext-Dev`. The page should look like this after you click on it:

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/SpacesTools.png">

4. For this demo, we'll use Cursor, but any [MCP client](https://github.com/punkpeye/awesome-mcp-clients) should follow a similar procedure. Scroll back up to the top of [MCP settings](https://huggingface.co/settings/mcp) page, and click on the `Cursor` icon of the `Setup with your AI assistant` section. Now, copy that code snippet and place it in your cursor settings file.

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/CursorScreenshot.png">

5. Now, when you start a new chat session in cursor you can ask it to edit an image! Note that for now the image must be available via a public URL. You can create a [Hugging Face Dataset](https://huggingface.co/datasets) to store your images online.

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/FluxKontextDevMcp.png">

> [!TIP]
> Using a popular public space as a tool may mean you have to wait longer to receive results. If you visit the space, you can click "Duplicate This Space" to create a private version of the space for yourself. If the space is using "ZeroGPU", you may need to update to a [PRO](https://huggingface.co/settings/billing/subscription) account to duplicate it.

6. Bonus: You can also search for MCP-compatible spaces with the Hugging Face MCP server! After completing step 4, you can also ask your LLM to find spaces that accomplish a certain task:

<img src="https://huggingface.co/datasets/freddyaboulton/bucket/resolve/main/SpacesSearch.png">

## Conclusion

This blog post has walked you through the exciting new capabilities that the Model Context Protocol (MCP) brings to Large Language Models. We've seen how Gradio apps, particularly those hosted on Hugging Face Spaces, are now fully MCP compliant, effectively turning Spaces into a vibrant "App Store" for LLM tools. By connecting these specialized MCP servers, your LLM can transcend basic question-answering and gain powerful new abilities, from image editing to transcription, to anything you can imagine!
