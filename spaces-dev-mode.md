---
title: "Introducing Spaces Dev Mode for a seamless developer experience" 
thumbnail: /blog/assets/spaces-dev-mode/thumbnail.jpg
authors:
- user: pagezyhf
---

# Introducing Spaces Dev Mode for a seamless developer experience

Hugging Face Spaces makes it easy for you to create and deploy AI-powered demos in minutes. Over 500,000 Spaces have been created by the Hugging Face community and it keeps growing! As part of [Hugging Face Spaces](https://huggingface.co/spaces), we recently released support for “Dev Mode”, to make your experience of building Spaces even more seamless.

Spaces Dev Mode lets you connect with VS Code or SSH directly to your Space. In a click, you can connect to your Space, and start editing your code, removing the need to push your local changes to the Space repository using git.
Let's see how to setup this feature in your Space’s settings 🔥

## Enable Dev Mode

Spaces Dev Mode is currently in beta, and available to [PRO subscribers](https://huggingface.co/pricing#pro). To learn more about Spaces Dev Mode, check out the [documentation](https://huggingface.co/dev-mode-explorers). After creating your space, navigate to Settings.

![dev-mode-settings-1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/spaces-dev-mode/dev-mode-settings-1.png)

Scroll down in the Settings and click on “Enable Dev Mode”. Your Space will automatically Restart. 

![dev-mode-settings-2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/spaces-dev-mode/dev-mode-settings-2.png)

## Connect to VS Code

Once your Space is in a Running state, you can connect to VS Code locally or in your browser in one click! You can also use SSH to set up the connection to your Space in another IDE.

![dev-mode-connect](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/spaces-dev-mode/dev-mode-connect.png)

For example, let’s change the color theme of this Gradio Space. After editing the code, no need to push your changes and rebuild the Space container to test it. Go directly in your Space and click “Refresh”.

![dev-mode-refresh](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/spaces-dev-mode/dev-mode-refresh.png)

That’s it! Once you’re satisfied with your changes, you can commit and merge to persist them.

![dev-mode-update](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/spaces-dev-mode/dev-mode-update.png)

Go build your first Spaces [here](https://huggingface.co/spaces)!