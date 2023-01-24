# The Hugging Face Blog Repository 🤗
This is the official repository of the [Hugging Face Blog](https://hf.co/blog). 

## How to write an article? 📝
1️⃣ Create a branch `YourName/Title`

2️⃣ Create a md (markdown) file, **use a short file name**.
For instance, if your title is "Introduction to Deep Reinforcement Learning", the md file name could be `intro-rl.md`. This is important because the **file name will be the blogpost's URL**.

3️⃣ Create a new folder in `assets`. Use the same name as the name of the md file. Optionally you may add a numerical prefix to that folder, using the number that hasn't been used yet. But this is no longer required. i.e. the asset folder in this example will be `123_intro-rl` or `intro-rl`. This folder will contain **your thumbnail only**. The folder number is mostly for (rough) ordering purposes, so it's no big deal if two concurrent articles use the same number.

For the rest of your files, create a mirrored folder in the HuggingFace Documentation Images [repo](https://huggingface.co/datasets/huggingface/documentation-images/tree/main/blog). This is to reduce bloat in the GitHub base repo when cloning and pulling.

🖼️: In terms of images, **try to have small files** to avoid having a slow loading user experience:
- Use compressed images, you can use this website: https://www.iloveimg.com/compress-image

4️⃣ Copy and paste this to your md file and change the elements
- title
- thumbnail
- Published (change the date)
- Change the author card
  - href ="/ your huggingface username"
  - src : your huggingface picture, for that right click to the huggingface picture and copy the link
  - <span class="fullname"> : your name

```
---
title: "PUT YOUR TITLE HERE" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
---

# Train your first Decision Transformer

<div class="blog-metadata">
    <small>Published September 02, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/decision-transformers-train.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/edbeeching"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1644220542819-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>edbeeching</code>
            <span class="fullname">Edward Beeching</span>
        </div>
    </a>
    <a href="/ThomasSimonini"> 
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1632748593235-60cae820b1c79a3e4b436664.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>ThomasSimonini</code>
            <span class="fullname">Thomas Simonini</span>
        </div>
    </a>
</div>
```

5️⃣ Then, you can add your content. It's markdown system so if you wrote your text on notion just control shift v to copy/paste as markdown.

6️⃣ Modify `_blog.yml` to add your blogpost.

7️⃣ When your article is ready, **open a pull request**.

8️⃣ The article will be **published automatically when you merge your pull request**.

## How to get a responsive thumbnail?
1️⃣ Create a `1300x650` image 

2️⃣ Use [this template](https://github.com/huggingface/blog/blob/main/assets/thumbnail-template.svg) and fill the content part.


## Using LaTeX

Just add:

```
\\(your_latex_here\\)
```

For instance:


``` \\( Q(S_t, A_t) \\) ``` ➡️ $Q(S_t, A_t)$
