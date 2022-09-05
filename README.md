# The Hugging Face Blog Repository 🤗
This is the official repository of the [Hugging Face Blog](hf.co/blog). 

## How to write an article? 📝
1️⃣ Create a branch `YourName/Title`

2️⃣ Create a md (markdown) file, try to be short.
For instance if your title is "Introduction to Deep Reinforcement Learning" the md file name will be `intro-rl.md`

3️⃣ Create a new folder in `assets` check the last folder number and name your folder `number_md-name` for instance `101_intro-rl`, this folder will contain all your illustrations and thumbnail.

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

5️⃣ Then, you can add your text, it's markdown system so if you wrote your text on notion just control shift v to copy/paste as markdown.

6️⃣ Modify `_blog.yml` to add your blogpost.

7️⃣ When your article is ready, open a pull Request.

8️⃣ The article will be published when you merge your pull request.

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
