---
title: 'Welcome CleanRL to the Hugging Face Hub 🤗'
thumbnail: /blog/assets/114_cleanrl/thumbnail.png
---

<h1>
    Welcome CleanRL to the Hugging Face Hub 🤗
</h1>

<div class="blog-metadata">
    <small>Published November 27, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/sample-factory.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/ThomasSimonini"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1632748593235-60cae820b1c79a3e4b436664.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>ThomasSimonini</code>
            <span class="fullname">Thomas Simonini</span>
        </div>
    </a>
</div>

At Hugging Face, we are contributing to the ecosystem for Deep Reinforcement Learning researchers and enthusiasts. That’s why we’re happy to announce that we integrated [CleanRL](https://github.com/vwxyzjn/cleanrl) to the Hugging Face Hub.

[CleanRL](https://github.com/vwxyzjn/cleanrl) is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features.
With this integration, you can now host your saved models 💾 and load powerful models from the community.

ADD VIDEO CleanRL

In this article, we’re going to show how you can do it. But if you want to follow with a tutorial, we wrote one where you'll learn:
- How to train a Deep Reinforcement Learning lander agent to land correctly on the Moon 🌕 
- How to upload it to the Hub 🚀

![gif](assets/47_sb3/lunarlander.gif)

- How to download and use a saved model from the Hub that CHOOSE ENVIRONMENT

👉 [The tutorial]() ADD LINK

Sounds exciting? Let's get started!

- [Install the library]()
- [Exploring CleanRL in the Hub]()
- [Uploading models to the Hub]()
- [Loading models from the Hub]()

TODO finish the table des matières when titles are chosen


## What's next?
In the coming weeks and months, we plan on supporting other tools from the ecosystem:

- Integrating [Sample Factory](https://github.com/alex-petrenko/sample-factory) to the Hub.
- Launching the Beta of [Simulate](https://github.com/huggingface/simulate), a library for easily creating and sharing simulation environments for intelligent agents
- Expanding our repository of Decision Transformer models with models trained or finetuned in an online setting [1]

The best way to keep in touch is to **[join our discord server](https://discord.gg/YRAq8fMnUG)** to exchange with us and with the community.

Finally, we would like to thank the ADD COSTA THANKS

### Would you like to integrate your library to the Hub?

This integration is possible thanks to the [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library which has all our widgets and the API for all our supported libraries. If you would like to integrate your library to the Hub, we have a [guide](https://huggingface.co/docs/hub/models-adding-libraries) for you!

## References
[1] Zheng, Qinqing and Zhang, Amy and Grover, Aditya “Online Decision Transformer” (arXiv preprint, 2022)
