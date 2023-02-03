# The Hugging Face Blog Repository 🤗
This is the official repository of the [Hugging Face Blog](https://hf.co/blog). 

## How to write an article? 📝
1️⃣ Create a branch `YourName/Title`

2️⃣ Create a md (markdown) file, **use a short file name**.
For instance, if your title is "Introduction to Deep Reinforcement Learning", the md file name could be `intro-rl.md`. This is important because the **file name will be the blogpost's URL**.

3️⃣ Create a new folder in `assets`. Use the same name as the name of the md file. Optionally you may add a numerical prefix to that folder, using the number that hasn't been used yet. But this is no longer required. i.e. the asset folder in this example will be `123_intro-rl` or `intro-rl`. This folder will contain **your thumbnail only**. The folder number is mostly for (rough) ordering purposes, so it's no big deal if two concurrent articles use the same number.

For the rest of your files, create a mirrored folder in the HuggingFace Documentation Images [repo](https://huggingface.co/datasets/huggingface/documentation-images/tree/main/blog). This is to reduce bloat in the GitHub base repo when cloning and pulling.

🖼️: In terms of images, **try to have small files** to avoid having a slow loading user experience:
- Use compressed images, you can use this website: https://tinypng.com or https://www.iloveimg.com/compress-image

4️⃣ Copy and paste this to your md file and change the elements
- title
- thumbnail
- authors
```
---
title: "PUT YOUR TITLE HERE" 
thumbnail: /blog/assets/101_decision-transformers-train/thumbnail.gif
authors:
- user: your_hf_user
- user: your_coauthor
---

# Train your first Decision Transformer

{blog_metadata}
{authors}
```

5️⃣ Then, you can add your content. It's markdown system so if you wrote your text on notion just control shift v to copy/paste as markdown.

6️⃣ Modify `_blog.yml` to add your blogpost.

7️⃣ When your article is ready, **open a pull request**.

8️⃣ The article will be **published automatically when you merge your pull request**.

## How to get a responsive thumbnail?
1️⃣ Create a `1300x650` image 

2️⃣ Use [this template](https://github.com/huggingface/blog/blob/main/assets/thumbnail-template.svg) and fill the content part.

➡️ Or select a background you like and follow the instructions in [this Figma template](https://www.figma.com/file/sXrf9VtkkbWI7kCIesMkDY/HF-Blog-Template?node-id=351%3A39).


## Using LaTeX

Just add:

```
\\(your_latex_here\\)
```

For instance:


``` \\( Q(S_t, A_t) \\) ``` ➡️ $Q(S_t, A_t)$
