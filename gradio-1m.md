---
title: "Journey to 1 Million Gradio Users!"
thumbnail: /blog/assets/gradio-1m/thumbnail.png
authors:
- user: abidlabs
---

# Journey to 1 Million Gradio Users!

5 years ago, we launched Gradio as a simple Python library to let researchers at Stanford easily demo computer vision models with a web interface. 

Today, Gradio is used by >1 million developers each month to build and share AI web apps. This includes some of the most popular open-source projects of all time, like [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui), [Oobabooga’s Text Generation WebUI](https://github.com/oobabooga/text-generation-webui), [Dall-E Mini](https://huggingface.co/spaces/dalle-mini/dalle-mini), and [LLaMA-Factory](https://huggingface.co/spaces/hiyouga/LLaMA-Board). 

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/maus.png)

How did we get here? How did Gradio keep growing in the very crowded field of open-source Python libraries? I get this question a lot from folks who are building their own open-source libraries. This post distills some of the lessons that I have learned over the past few years:

1. Invest in good primitives, not high-level abstractions
2. Embed virality directly into your library
3. Focus on a (growing) niche
4. Your only roadmap should be rapid iteration
5. Maximize ways users can consume your library's outputs

-----

**1. Invest in good primitives, not high-level abstractions**

When we first launched Gradio, we offered only one high-level class (`gr.Interface`), which created a complete web app from a single Python function. We quickly realized that developers wanted to create other kinds of apps (e.g. multi-step workflows, chatbots, streaming applications), but as we started listing out the apps users wanted to build, we realized what we needed to do: go lower.

Instead of building many high-level classes to support different use cases, we built a low-level API called Gradio Blocks that let users assemble applications from modular components, events, and layouts. Despite generally being more work to use, `gr.Blocks` today represents 80% of the usage of Gradio—including the highly popular apps mentioned above.

With a small team, a focus on low-level abstractions has necessarily meant that we could not chase many tempting, high-level abstractions. But this focus saved us from two common pitfalls:

* **Customization-maintenance trap**: high-level abstractions are easy to use, but users request extra parameters to customize them, which in turn leads to increased maintenance burden since every abstraction and parameter needs to be implemented, maintained, and tested to avoid bugs. 

* **The productivity illusion**: using high-level abstractions seems like less work, until a user needs functionality that is not supported (which can be hard to predict at the beginning of a project), forcing the developer to do a costly rewrite.

The first problem wastes _our time_ as maintainers of Gradio, while the second problem wastes _our users’ time_. To this day, Gradio only includes two high-level abstractions (`gr.Interface` and `gr.ChatInterface`) which themselves are built with Blocks.

The advantage of having good primitives is even more dramatic in the age of AI-assisted coding, where we find that LLMs are generally good at combing through documentation and building complex applications out of primitives (see all the games being built with `three.js` for example). It doesn’t take much extra time to write low-level code if the AI is writing the code for you. 

**2. Focus on a (growing) niche**

The best ambassador for your library is not you, but an enthusiastic user. It follows, then, that you should find ways for users to share your library or its products as part of their workflow. Gradio’s early growth was fueled by our “share links” feature (which created a temporary public link for your Gradio app in a single line of code). Instead of having to worry about packaging or hosting their code on a web server with the right kind of compute, Gradio users could share their apps with colleagues immediately.

After joining Hugging Face, we also benefited from being the standard UI for Hugging Face Spaces—widely used by the machine learning research community. The most viral Spaces drew millions of visitors, exposing developers to Gradio, who in turn shared and built their own Gradio apps, using those same Spaces (whose code was publicly available) as a resource to learn how to use Gradio.

**3. Build for a (growing) niche use case**

Early on, we faced a critical decision: should Gradio be a general-purpose Python web framework, or should we focus specifically on building machine learning web apps? We chose the latter, and this has made all the difference.

In the crowded ecosystem of Python web libraries, we are frequently asked, "How is Gradio different from X?" Our concise answer—that Gradio is optimized for machine learning web apps—is memorable and fundamentally accurate. Each Gradio app ships with features particularly suited to ML workflows, such as a built-in queue that can manage long-running ML tasks efficiently, even with thousands of concurrent users. The components we have designed are specifically tailored to ML use cases. On the other hand, for a long time, Gradio did not even include features like link buttons, simply because our core users never needed it. By narrowing our focus, Gradio quickly became the go-to choice among machine learning developers—the default "UI for AI."

Of course, we benefited tremendously from choosing a niche that itself was growing. The recent headwinds that have propelled all things AI-related benefited us tremendously and we likely would not have experienced the same kind of growth if we had focused on data science or dashboards for example.

**4. Your only roadmap should be rapid iteration**

Unlike some other libraries, Gradio doesn’t publish roadmaps. Instead, we track emerging trends in machine learning and ship accordingly. In open-source software generally, and AI specifically, shipping features based on the needs of the community (and deprecating features that are no longer needed) is key to continuous growth. As a concrete example, in the early versions of Gradio we built specific functionality to allow developers to show the “interpretation” of their machine learning models—as this was a very popular use case around 2020-21. As interest in interpretation waned, we deprecated this and instead poured our efforts into audio/video streaming and chat-related features. 

Our internal process is decentralized as well. Each of the 11 engineers and developer advocates on the Gradio team is encouraged to identify impactful ideas, prototype quickly, engage directly with the open-source community via GitHub, Hugging Face, and social media. Each team member brings these ideas back to the team and we continuously build and rebuild consensus around what ideas are likely to be impactful as we keep growing the most impactful directions. 

**5. Maximize ways users can consume your library's outputs**

When you create a Gradio app and “launch()” it, you get a web application running in your browser. But that’s not all -- you also get an API endpoint for each Python function, as well as automatically-generated documentation for each endpoint. These endpoints can be consumed via Python or JavaScript Clients that our team has built, or directly using cURL.

The reason that we do this is because we want each minute that a developer spends with Gradio to be maximally useful. A single Gradio app should run locally or be deployed without any code changes on Hugging Face Spaces or on your server, or integrate into larger applications programmatically, or even be harnessed by AI agents through MCPs (more on that soon!). By focusing on maximum usability, we hope that developers can continue to get more and more out of the Gradio library. 

**Onwards to 10 million!**

