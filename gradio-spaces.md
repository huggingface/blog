---
title: "Showcase Your Projects in Spaces using Gradio"
thumbnail: /blog/assets/27_gradio-spaces/thumbnail.png

---

<h1>
    Showcase Your Projects in Spaces using Gradio
</h1>

<div class="blog-metadata">
    <small>Published September 21, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/gradio-spaces.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/merve">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1631694399207-6141a88b3a0ec78603c9e784.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>merve</code>
            <span class="fullname">Merve Noyan</span>
        </div>
    </a>
</div>




It's so easy to demonstrate a machine learning project thanks to [Gradio](https://gradio.app/). In this blog post, we'll walk you through the recent Gradio integration that helps you wrap your Hugging Face model with Gradio seamlessly with few lines of code, and how to host your model checkpoints that are not in the Hugging Face hub, in Hugging Face Spaces.



## Hugging Face Hub Integration in Gradio

You can demonstrate your models in Hub easily. You only need to define the [Interface](https://gradio.app/docs#interface) that will include the model name you want to infer with, define additional descriptions and titles, maybe an example input to guide your audience. After defining your Interface, just call `.launch()` and serve your model. You can directly put this in Spaces as well!

```
import gradio as gr

description = "Story generation with GPT-2"
title = "Generate your own story 

interface = gr.Interface.load("huggingface/pranavpsv/gpt2-genre-story-generator",
    description=description,
    examples=[["Adventurer is approached by a mysterious stranger in the tavern for a new quest."]])

interface.launch()
```


![story-gen](assets/27_gradio-spaces/story-gen.png)

This integration supports different types of models, `image-to-text`, `speech-to-text`, `text-to-speech` and more.

![big-gan](assets/27_gradio-spaces/big-gan.png)

You can run your demos anywhere, you can put this in Spaces as well by simply committing your `app.py` file. This integration is built on top of Hugging Face pipelines, so please make sure your model supports [pipelines](https://huggingface.co/transformers/main_classes/pipelines.html). But don't worry, we'll cover different ways of demonstrating models.

## Serving Custom Model Checkpoints with Gradio in Hugging Face Spaces
You can serve your models in Spaces even if your model does not have a defined pipeline or is not hosted in the Hub. Just wrap your application in Gradio Interface as described above and put it in Spaces. 
![imagenet-demo](assets/27_gradio-spaces/imagenet-demo.gif)

## Mix and Match Models!

Using Gradio Series, you can mix-and-match different models! Here, we've put French to English translation model on top of the story generator and English to French translation model at the end of the generator model to simply make a French story generator.

```
import gradio as gr
from gradio.mix import Series

description = "Generate your own D&D story!"
title = "French Story Generator using Opus MT and GPT-2"

translator_fr = gr.Interface.load("huggingface/Helsinki-NLP/opus-mt-fr-en")
story_gen = gr.Interface.load("huggingface/pranavpsv/gpt2-genre-story-generator")
translator_en = gr.Interface.load("huggingface/Helsinki-NLP/opus-mt-en-fr")

Series(translator_fr, story_gen, translator_en, description = description,
    title = title, examples=[["L'aventurier est approché par un mystérieux étranger, pour une nouvelle quête."]], inputs = gr.inputs.Textbox(lines = 10)).launch()

```

![story-gen-fr](assets/27_gradio-spaces/story-gen-fr.png)

## Uploading your Models to the Spaces

You can serve your demos in Hugging Face thanks to Spaces! To do this, simply create a new Space, and then drag and drop your demos or use Git. 

![spaces-demo](assets/27_gradio-spaces/spaces-demo-finalized.gif)

Easily build your first demo with Spaces [here](https://huggingface.co/spaces)!