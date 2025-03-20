---
title: "Open R1: How to use OlympicCoder locally for coding"
thumbnail: /blog/assets/olympic-coder-lmstudio/banner.png
authors:
- user: burtenshaw
- user: reach-vb
- user: lewtun
- user: edbeeching
- user: yagilb
  guest: true
  org: lmstudio-ai
---

# Open R1: How to use OlympicCoder locally for coding

Everyone’s been using Claude and OpenAI as coding assistants for the last few years, but there’s less appeal if you look at the developments coming out of open source projects like [Open R1](https://huggingface.co/open-r1). If we look at the evaluation on [LiveCodeBench](https://livecodebench.github.io) below, we can see that the 7B parameter variant outperforms Claude 3.7 Sonnet and GPT-4o. These models are the daily driver of many engineers in applications like Cursor and VSCode.

![evals](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/olympic-coder-lmstudio/lcb-evals.png)

Evals are great and all, but I want to get my hands dirty and feel the commits\! This blog post focuses on how you can integrate these models in your IDE now. We will set up OlympicCoder 7B, the smaller of the two OlympicCoder variants, and we’ll use a quantized variant for optimum local inference. Here’s the stack we’re going to use:

* OlympicCoder 7B. The 4bit GGUF version from the [LMStudio Community](https://huggingface.co/lmstudio-community/OlympicCoder-7B-GGUF)  
* LM Studio: A tool that simplifies running AI models  
* Visual Studio Code (VS Code)  
* [Continue](https://www.continue.dev/) a VS Code extension for local models

It’s important to say that we chose this stack purely for simplicity. You might want to experiment with the larger model and/ or different GGUF files. Or even alternative inference engines like [llama.cpp](https://github.com/ggml-org/llama.cpp). 

![generation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/olympic-coder-lmstudio/generation.gif)

# 1\. Install LM Studio

LM Studio is like a control panel for AI models. It integrates with the Hugging Face hub to pull models, helps you find the right GGUF file, and exposes an API that other applications can use to interact with the model.

In short, it lets you download and run them without any complicated setup.

1. Go to the LM Studio website: Open your web browser and go to [https://lmstudio.ai/download](https://lmstudio.ai/download).  
2. Choose your operating system: Click the download button for your computer (Windows, Mac, or Linux).  
3. Install LM Studio: Run the downloaded file and follow the instructions. It’s just like installing any other program.

# 2\. Get [OlympicCoder 7B](https://huggingface.co/open-r1/OlympicCoder-7B)

The GGUF files that we need are hosted on the hub. We can open the model from the hub in LMStudio, using the ‘Use this model’ button:

![model_page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/olympic-coder-lmstudio/model_page.png)

This will link to the LMStudio application and open it on your machine. You’ll just need to Choose a Quantization. I went for `Q4_K_M` because it will perform well on most devices. If you have more compute, you might want to try out one of the options with `Q8_*`.

If you want to skip the UI, you can also load models with `LMStudio` via the command line:

```
lms get lmstudio-community/OlympicCoder-7B-GGUF
lms load olympiccoder-7b
lms server start
```

# 3\. Connect LM Studio to VS Code

This is the important part. We now need to integrate VScode with the model served by LMStudio.

1. In LM Studio, activate the server on the ‘Developer’ tab. This will expose the endpoints at `http://localhost:1234/v1`.

![lmstudio](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/olympic-coder-lmstudio/lm-studio.png)

2. **Install the VS Code Extension** to connect to our local server. I went for Continue.dev, but there are other options too.  
   * In VSCode, go to the Extensions view (click the square icon on the left sidebar, or press Ctrl+Shift+X / Cmd+Shift+X).  
   * Search for “Continue” and install the extension from “Continue Dev”.  
3. **Configure a New Model** in Continue.dev  
   * Open the Continue tab and in the models dropdown, select ‘add new chat model’.  
   * This will open a json configuration file. You’ll need to specify the model name. I.e. olympiccoder-7b

![continue](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/olympic-coder-lmstudio/continue_dev.png)

# 🚀 You’ve got a local coding assistant\!

Most of the core AI features in vscode are available via this setup, for example:

* **Code Completion:** Start typing, and the AI will suggest how to finish your code.  
* **Generate Code:** Ask it to write a function or a whole block of code. For example, you could type (in a comment or a chat window, depending on the extension): // Write a function to reverse a string in JavaScript  
* **Explain Code:** Select some code and ask the AI to explain what it does.  
* **Refactor Code:** Ask the AI to make your code cleaner or more efficient.  
* **Write Tests:** Ask the AI to create unit tests for your code.

# 🏋️‍♀️ What’s the vibe of OlympicCoder?

OlympicCoder is not Claude. It’s optimised on the [CodeForces-CoTs](https://huggingface.co/datasets/open-r1/codeforces-cots) dataset which is based on competitive coding challenges. That means that you should not expect it to be super friendly and explanatory. Instead, roll up your sleeves and expect a no-holds barred competitive coder ready to deal with tough problems. 

You might want to mix up OlympicCoder with other models to get a rounded coding experience. For example, if you’re trying to squeeze milliseconds out of a binary search, try OlympicCoder. If you want to design a user facing API, go for Claude-3.7-sonnet or [Qwen-2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct).

# Next Steps

- Share your favorite generations in the comments below  
- Try out another variant of [OlympicCoder](https://huggingface.co/collections/open-r1/olympiccoder-67d0927b5ee0dde083bed8cd) from the hub.  
- Experiment with quantization types based on your hardware.   
- Try out multiple models in LM Studio for different coding vibes\! Check out the model catalog [https://lmstudio.ai/models](https://lmstudio.ai/models)  
- Experiment with other VS Code extensions like [Cline](https://github.com/cline/cline) which have agentic functionality