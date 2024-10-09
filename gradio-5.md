---
title: "Welcome, Gradio 5" 
thumbnail: /blog/assets/gradio-5/thumbnail.png
authors:
- user: abidlabs
---

# Welcome, Gradio 5

We’ve been hard at work over the past few months, and we are excited to now announce the **stable release of Gradio 5**. 

With Gradio 5, developers can build **production-ready machine learning web applications** that are performant, scalable, beautifully designed, accessible, and follow best web security practices, all in a few lines of Python.

To give Gradio 5 a spin, simply type in your terminal:

```
pip install --upgrade gradio
```

and start building your [first Gradio application](https://www.gradio.app/guides/quickstart).

## Gradio 5: production-ready machine learning apps

If you have used Gradio before, you might be wondering what’s different about Gradio 5. 

Our goal with Gradio 5 was to listen to and address the most common pain points that we’ve heard from Gradio developers about building production-ready Gradio apps. For example, we’ve heard some developers tell us:

*   “Gradio apps load too slowly” → Gradio 5 ships with major performance improvements, including the ability to serve Gradio apps via server-side rendering (SSR) which loads Gradio apps almost instantaneously in the browser. _No more loading spinner_!  🏎️💨

<video width="600" controls playsinline>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-5/gradio-4-vs-5-load.mp4">
</video>


*   "This Gradio app looks old-school" → Many of the core Gradio components, including Buttons, Tabs, Sliders, as well as the high-level chatbot interface, have been refreshed with a more modern design in Gradio 5. We’re also releasing a new set of built-in themes, to let you easily create fresh-looking Gradio apps 🎨

*   “I can’t build realtime apps in Gradio” → We have unlocked low-latency streaming in Gradio! We use base64 encoding and websockets automatically where it offers speedups, support WebRTC via custom components, and have also added significantly more documentation and example demos that are focused on common streaming use cases, such as webcam-based object detection, video streaming, real-time speech transcription and generation, and conversational chatbots. 🎤


*   "LLMs don't know Gradio" → Gradio 5 ships with an experimental AI Playground where you can use AI to generate or modify Gradio apps and preview the app right in your browser immediately: [https://www.gradio.app/playground](https://www.gradio.app/playground) 

<video width="600" controls>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/blog/gradio-5/simple-playground.mp4">
</video>

Gradio 5 provides all these features while maintaining Gradio’s simple and intuitive developer-facing API. Since Gradio 5 intended to be a production-ready web framework for all kinds of machine learning applications, we've also made significant improvements around web security (including getting a 3rd-party audit of Gradio) -- more about that in an upcoming post!

## Breaking changes

Gradio apps that did not raise any deprecation warnings in Gradio 4.x should continue to work in Gradio 5, with a [small number of exceptions. See a list of breaking changes in Gradio 5 here](https://github.com/gradio-app/gradio/issues/9463). 

## What’s next for Gradio?

Many of the changes we’ve made in Gradio 5 are designed to enable new functionality that we will be shipping in the coming weeks. Stay tuned for:

*   Multi-page Gradio apps, along with native navbars and sidebars
    
*   Support for running Gradio apps on mobile using PWA and potentially native app support
    
*   More media components to support emerging modalities around images and videos
    
*   A richer DataFrame component with support for common spreadsheet-type operations

*   One-line integrations with machine learning model and API providers
    
*   Further improvements that will decrease the memory consumption of Gradio apps

And much more! With Gradio 5 providing a robust foundation to build web applications, we’re excited to _really_ _get started_ letting developers build all sorts of ML apps with Gradio.

## Try Gradio 5 right now

Here are some Hugging Face Spaces that are running Gradio 5:

* https://huggingface.co/spaces/akhaliq/depth-pro
* https://huggingface.co/spaces/hf-audio/whisper-large-v3-turbo
* https://huggingface.co/spaces/gradio/chatbot_streaming_main
* https://huggingface.co/spaces/gradio/scatter_plot_demo_main