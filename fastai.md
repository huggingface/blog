---
title: 'Welcome fastai to the Hugging Face Hub'
thumbnail: /blog/assets/64_fastai/fastai_hf_blog.png
---

<h1>
    Welcome fastai to the Hugging Face Hub
</h1>

<div class="blog-metadata">
    <small>Published May 6, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/fastai.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/espejelomar"> 
        <img class="avatar avatar-user" src="https://bafybeidj6oxo7zm5pejnc2iezy24npw4qbt2jgpo4n6igt7oykc7rbvcxi.ipfs.dweb.link/omar_picture.png" title="Gravatar">
        <div class="bfc">
            <code>espejelomar</code>
            <span class="fullname">Omar Espejel</span>
        </div>
    </a>
</div>

## Making neural nets uncool again... and sharing them

<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/64_fastai_hub.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

> **Update**: May 17, 2022, to (1) comply with version [0.6.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.6.0) of the `huggingface_hub` library; and (2) to add a `blurr` library example.

Few have done as much as the [fast.ai](https://www.fast.ai/) ecosystem to make Deep Learning accessible. Our mission at Hugging Face is to democratize good Machine Learning. Let's make exclusivity in access to Machine Learning, including [pre-trained models](https://huggingface.co/models), a thing of the past and let's push this amazing field even further.

fastai is an [open-source Deep Learning library](https://github.com/fastai/fastai) that leverages PyTorch and Python to provide high-level components to train fast and accurate neural networks with state-of-the-art outputs on text, vision, and tabular data. However, fast.ai, the company, is more than just a library; it has grown into a thriving ecosystem of open source contributors and people learning about neural networks. As some examples, check out their [book](https://github.com/fastai/fastbook) and [courses](https://course.fast.ai/). Join the fast.ai [Discord](https://discord.com/invite/YKrxeNn) and [forums](https://forums.fast.ai/). It is a guarantee that you will learn by being part of their community!

Because of all this, and more (the writer of this post started his journey thanks to the fast.ai course), we are proud to announce that fastai practitioners can now share and upload models to Hugging Face Hub with a single line of Python.

 👉 In this post, we will introduce the integration between fastai and the Hub. Additionally, you can open this tutorial as a [Colab notebook](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/64_fastai_hub.ipynb).

We want to thank the fast.ai community, notably [Jeremy Howard](https://twitter.com/jeremyphoward), [Wayde Gilliam](https://twitter.com/waydegilliam), and [Zach Mueller](https://twitter.com/TheZachMueller) for their feedback 🤗. This blog is heavily inspired by the [Hugging Face Hub section](https://docs.fast.ai/huggingface.html) in the fastai docs.


## Why share to the Hub?

The Hub is a central platform where anyone can share and explore models, datasets, and ML demos. It has the most extensive collection of Open Source models, datasets, and demos.

Sharing on the Hub amplifies the impact of your fastai models by making them available for others to download and explore. You can also use transfer learning with fastai models; load someone else's model as the basis for your task.

Anyone can access all the fastai models in the Hub by filtering the [hf.co/models](https://huggingface.co/models?library=fastai&sort=downloads) webpage by the fastai library, as in the image below.

![Fastai Models in the Hub](assets/64_fastai/hf_hub_fastai.png)

In addition to free model hosting and exposure to the broader community, the Hub has built-in [version control based on git](https://huggingface.co/docs/transformers/model_sharing#repository-features) (git-lfs, for large files) and [model cards](https://huggingface.co/docs/hub/model-repos#what-are-model-cards-and-why-are-they-useful) for discoverability and reproducibility. For more information on navigating the Hub, see [this introduction](https://github.com/huggingface/education-toolkit/blob/main/01_huggingface-hub-tour.md).



## Joining Hugging Face and installation

To share models in the Hub, you will need to have a user. Create it on the [Hugging Face website](https://huggingface.co/join).

The `huggingface_hub` library is a lightweight Python client with utility functions to interact with the Hugging Face Hub. To push fastai models to the hub, you need to have some libraries pre-installed (fastai>=2.4, fastcore>=1.3.27 and toml). You can install them automatically by specifying ["fastai"] when installing `huggingface_hub`, and your environment is good to go:

```bash
pip install huggingface_hub["fastai"]
```

## Creating a fastai `Learner`

Here we train the [first model in the fastbook](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb) to identify cats 🐱. We fully recommended reading the entire fastbook.

```py
# Training of 6 lines in chapter 1 of the fastbook.
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```


## Sharing a `Learner` to the Hub

A [`Learner` is a fastai object](https://docs.fast.ai/learner.html#Learner) that bundles a model, data loaders, and a loss function. We will use the words `Learner` and Model interchangeably throughout this post.

First, log in to the Hugging Face Hub. You will need to create a `write` token in your [Account Settings](http://hf.co/settings/tokens). Then there are three options to log in:

1. Type `huggingface-cli login` in your terminal and enter your token.

2. If in a python notebook, you can use `notebook_login`.

```py
from huggingface_hub import notebook_login

notebook_login()
```

3. Use the `token` argument of the `push_to_hub_fastai` function.

You can input `push_to_hub_fastai` with the `Learner` you want to upload and the repository id for the Hub in the format of "namespace/repo_name". The namespace can be an individual account or an organization you have write access to (for example, 'fastai/stanza-de'). For more details, refer to the [Hub Client documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.push_to_hub_fastai).

```py
from huggingface_hub import push_to_hub_fastai

# repo_id = "YOUR_USERNAME/YOUR_LEARNER_NAME"
repo_id = "espejelomar/identify-my-cat"

push_to_hub_fastai(learner=learn, repo_id=repo_id)
```

The `Learner` is now in the Hub in the repo named [`espejelomar/identify-my-cat`](https://huggingface.co/espejelomar/identify-my-cat). An automatic model card is created with some links and next steps. When uploading a fastai `Learner` (or any other model) to the Hub, it is helpful to edit its model card (image below) so that others better understand your work (refer to the [Hugging Face documentation](https://huggingface.co/docs/hub/model-repos#what-are-model-cards-and-why-are-they-useful)).

![Fastai Model Card](assets/64_fastai/hf_model_card.png)

if you want to learn more about `push_to_hub_fastai` go to the [Hub Client Documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_fastai). There are some cool arguments you might be interested in 👀. Remember, your model is a [Git repository](https://huggingface.co/docs/transformers/model_sharing#repository-features) with all the advantages that this entails: version control, commits, branches...

## Loading a `Learner` from the Hugging Face Hub

Loading a model from the Hub is even simpler. We will load our `Learner`, "espejelomar/identify-my-cat", and test it with a cat image (🦮?). This code is adapted from
the [first chapter of the fastbook](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb).

First, upload an image of a cat (or possibly a dog?). The [Colab notebook with this tutorial](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/64_fastai_hub.ipynb) uses `ipywidgets` to interactively upload a cat image (or not?). Here we will use this cute cat 🐅:

![Fastai Model Card](assets/64_fastai/cat.jpeg)

Now let's load the `Learner` we just shared in the Hub and test it.

```py
from huggingface_hub import from_pretrained_fastai

# repo_id = "YOUR_USERNAME/YOUR_LEARNER_NAME"
repo_id = "espejelomar/identify-my-cat"

learner = from_pretrained_fastai(repo_id)
```
It works 👇!

```py
_,_,probs = learner.predict(img)
print(f"Probability it's a cat: {100*probs[1].item():.2f}%")

Probability it's a cat: 100.00%
```

The [Hub Client documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_fastai) includes addtional details on `from_pretrained_fastai`.


## `Blurr` to mix fastai and Hugging Face Transformers (and share them)!

> [Blurr is] a library designed for fastai developers who want to train and deploy Hugging Face transformers - [Blurr Docs](https://github.com/ohmeow/blurr).

We will:
1. Train a `blurr` Learner with the [high-level Blurr API](https://github.com/ohmeow/blurr#using-the-high-level-blurr-api). It will load the `distilbert-base-uncased` model from the Hugging Face Hub and prepare a sequence classification model.
2. Share it to the Hub with the namespace `fastai/blurr_IMDB_distilbert_classification` using `push_to_hub_fastai`.
3. Load it with `from_pretrained_fastai` and try it with `learner_blurr.predict()`.

Collaboration and open-source are fantastic!

First, install `blurr` and train the Learner.

```bash
git clone https://github.com/ohmeow/blurr.git
cd blurr
pip install -e ".[dev]"
```

```python
import torch
import transformers
from fastai.text.all import *

from blurr.text.data.all import *
from blurr.text.modeling.all import *

path = untar_data(URLs.IMDB_SAMPLE)
model_path = Path("models")
imdb_df = pd.read_csv(path / "texts.csv")

learn_blurr = BlearnerForSequenceClassification.from_data(imdb_df, "distilbert-base-uncased", dl_kwargs={"bs": 4})
learn_blurr.fit_one_cycle(1, lr_max=1e-3)
```

Use `push_to_hub_fastai` to share with the Hub.

```python
from huggingface_hub import push_to_hub_fastai

# repo_id = "YOUR_USERNAME/YOUR_LEARNER_NAME"
repo_id = "fastai/blurr_IMDB_distilbert_classification"

push_to_hub_fastai(learn_blurr, repo_id)
```

Use `from_pretrained_fastai` to load a `blurr` model from the Hub.


```python
from huggingface_hub import from_pretrained_fastai

# repo_id = "YOUR_USERNAME/YOUR_LEARNER_NAME"
repo_id = "fastai/blurr_IMDB_distilbert_classification"

learner_blurr = from_pretrained_fastai(repo_id)
```

Try it with a couple sentences and review their sentiment (negative or positive) with `learner_blurr.predict()`.

```python
sentences = ["This integration is amazing!",
             "I hate this was not available before."]

probs = learner_blurr.predict(sentences)

print(f"Probability that sentence '{sentences[0]}' is negative is: {100*probs[0]['probs'][0]:.2f}%")
print(f"Probability that sentence '{sentences[1]}' is negative is: {100*probs[1]['probs'][0]:.2f}%")
```
Again, it works!

```python
Probability that sentence 'This integration is amazing!' is negative is: 29.46%
Probability that sentence 'I hate this was not available before.' is negative is: 70.04%
```


## What's next?

Take the [fast.ai course](https://course.fast.ai/) (a new version is coming soon), follow [Jeremy Howard](https://twitter.com/jeremyphoward?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) and [fast.ai](https://twitter.com/FastDotAI) on Twitter for updates, and start sharing your fastai models on the Hub 🤗. Or load one of the [models that are already in the Hub](https://huggingface.co/models?library=fastai&sort=downloads).

📧 Feel free to contact us via the [Hugging Face Discord](https://discord.gg/YRAq8fMnUG) and share if you have an idea for a project. We would love to hear your feedback 💖.


### Would you like to integrate your library to the Hub?

This integration is made possible by the [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) library. If you want to add your library to the Hub, we have a [guide](https://huggingface.co/docs/hub/adding-a-library) for you! Or simply tag someone from the Hugging Face team.

A shout out to the Hugging Face team for all the work on this integration, in particular [@osanseviero](https://twitter.com/osanseviero) 🦙.

Thank you fastlearners and hugging learners 🤗. 
