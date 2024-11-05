---
title: "Hugging Face + PyCharm"
thumbnail: /blog/assets/pycharm-integration/thumbnail.png
authors:
- user: rocketknight1
---

# Hugging Face + PyCharm

It’s a Tuesday morning. As a Transformers maintainer, I’m doing the same thing I do most weekday mornings: Opening [PyCharm](https://jb.gg/get-pycharm-hf), loading up the Transformers codebase and gazing lovingly at the [chat template documentation](https://huggingface.co/docs/transformers/main/chat_templating) while ignoring the 50 user issues I was pinged on that day. But this time, something feels different:

![screenshot 0](assets/pycharm_integration/screenshot_0.png)

Something is… wait\! Computer\! Enhance\!
![screenshot 1](assets/pycharm_integration/screenshot_1.png)  
Is that..?  
![screenshot 2](assets/pycharm_integration/screenshot_2.png)  
Those user issues are definitely not getting responses today. Let’s talk about the Hugging Face integration in PyCharm.

## The Hugging Face Is Inside Your House

I could introduce this integration by just listing features, but that’s boring and there’s [documentation](https://www.jetbrains.com/help/pycharm/hugging-face.html) for that. Instead, let’s walk through how we’d use it all in practice. Let’s say I’m writing a Python app, and I decide I want the app to be able to chat with users. Not just text chat, though – we want the users to be able to paste in images too, and for the app to naturally chat about them as well. 

If you’re not super-familiar with the current state-of-the-art in machine learning, this might seem like a terrifying demand, but don’t fear. Simply right click in your code, and select “Insert HF Model”. You’ll get a dialog box:
![dialog_box_screenshot](assets/pycharm_integration/dialog_box_screenshot.png)  
Chatting with both images and text is called “image-text-to-text” \- the user can supply images and text, and the model outputs text. Scroll down on the left until you find it. By default, the model list will be sorted by Likes \- but remember, older models often have a lot of likes built up even if they’re not really the state of the art anymore. We can check how old models are by seeing the date they were last updated, just under the model name. Let’s pick something that’s both recent and popular: microsoft/Phi-3.5-vision-instruct . 

You can select “Use Model” for some model categories to automatically paste some basic code into your notebook, but what often works better is to scroll through the Model Card on the right and grab any sample code. You can see the full model card to the right of the dialog box, exactly as it's shown on Hugging Face Hub. Let’s do that and paste it into our code\!
![code_snippet_screenshot](assets/pycharm_integration/code_snippet_screenshot.png)
Your office cybersecurity person might complain about you copying a chunk of random text from the internet and running it without even reading all of it, but if that happens just call them a nerd and continue regardless. And behold: We now have a working model that can happily chat about images \- in this case, it reads and comments on screenshots of a Microsoft slide deck. Feel free to play around with this example. Try your own chat, or your own images. Once you get it working, simply wrap this code into a class and it’s ready to go in your app. We just got state of the art open-source machine learning in ten minutes without even opening a web browser.

(Tip: These models can be large\! If you’re getting memory errors, try using a GPU with more memory, or try reducing the 20 in the sample code. You can also remove device\_map="cuda" to put the model in CPU memory instead, at the cost of speed)

## Instant Model Cards

Next, let’s change perspective in our little scenario. Now let’s say you’re not the author of this code \- you’re a coworker who has to review it. Maybe you’re the cybersecurity person from earlier, who’s still upset about the “nerd” comment. You look at this code snippet, and you have no idea what you’re seeing. Don’t panic \- just hover over the model name, and the entire model card instantly appears. You can quickly verify the origin of this model, and what its intended uses are. 

(This is also extremely helpful if you work on something else and completely forget everything about the code you wrote two weeks ago)

![model_card_screenshot](assets/pycharm_integration/model_card_screenshot.png)

## The Local Model Cache

You might notice that the model has to be downloaded the first time you run this code, but after that, it’s loaded much more quickly. The model has been stored in your local cache. Remember the mysterious little 🤗 icon from earlier? Simply click it, and you’ll get a listing of everything in your cache:
![model_cache_screenshot](assets/pycharm_integration/model_cache_screenshot.png)

This is a neat way to find the models you’re working with right now, and also to clear them out and save some disk space once you don’t need them anymore. It’s also very helpful for the two-week amnesia scenario \- if you can’t remember the model you were using back then, it’s probably in here. Remember, though \- most useful, production-ready models in 2024 are going to be \>1GB, so your cache can fill up fast\!

## Python in the age of AI

At Hugging Face, we tend to think of open-source AI as being a natural extension of the open-source philosophy: Open software solves problems for developers and users, creating new abilities for them to integrate into their code, and open models do the same. There is a tendency to be blinded by complexity, and to focus too much on the implementation details because they’re all so novel and exciting, but models exist to **do stuff for you.** If you abstract away the details of architecture and training, they’re fundamentally **functions** \- tools in your code that will transform a certain kind of input into a certain kind of output.

These features are a natural fit, as a result \- IDEs already automatically pull up function signatures and docstrings for you, and the AI equivalent of that is to pull up sample code and model cards for trained models. Integrations like these make it easy to reach over and import a chat or image recognition model as conveniently as you would import any other library. We think it’s obvious that this is what the future of code will look like, and we hope that you find these features useful as a result\!

[Download PyCharm](https://jb.gg/get-pycharm-hf) to give the Hugging Face integration a try.  
Get a free 3-month PyCharm subscription using the PyCharm4HF code [here](http://jetbrains.com/store/redeem/).