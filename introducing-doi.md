---
title: "Introducing DOI: the Digital Object Identifier to Datasets and Models"
thumbnail: /blog/assets/107_launching_doi/thumbnail.jpeg
---

  

# Introducing DOI: the Digital Object Identifier to Datasets and Models

<div class="blog-metadata">
    <small>Published October 7, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/introducing-doi.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
   <a href="/sasha">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1626198087984-60edd0133e2c73a9a21455f5.png?w=200&amp;h=200&amp;f=face" title="Gravatar">
        <div class="bfc">
            <code>sasha</code>
            <span class="fullname">Sasha Luccioni</span>
        </div>
    </a>
    <a href="/Sylvestre">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1665137450767-6258561f4d4291e8e63d8ae6.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>sylvestre</code>
            <span class="fullname">Sylvestre Bouchot</span>
        </div>
    </a>
    <a href="/cakiki">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1646492542174-5e70f6048ce3c604d78fe133.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>cakiki</code>
            <span class="fullname">Christopher Akiki</span>
            <span class="bg-gray-100 dark:bg-gray-700 rounded px-1 text-gray-600 text-sm font-mono">guest</span>
        </div>
   </a>
</div>

If you are part of the academic world, you have already asked yourself how to cite one of the datasets or models present on 🤗 Hub.

The purpose of our mission at Hugging Face is to democratize good machine learning. That includes best practices that make ML models and datasets more reproducible, better documented, and easier to use and share.

To solve this challenge, **we're excited to announce that you can now generate a DOI for your model or dataset directly from the Hub**!

<kbd>
  <img alt="Generating DOI" src="assets/107_launching_doi/doi.gif">
</kbd>

## DOIs in a nutshell and why do they matter?

DOIs (Digital Object Identifiers) are strings that can be used to uniquely identify a digital object, anything from articles to figures, including datasets and models. DOIs are tied to object metadata, including things like the object’s URL, version, date of creation, description, etc, that make it commonly accepted form of reference to digital resources across research and academic communities; it's analogous to a book's ISBN.
This makes it easier to both find information about an object, but also to share objects with the world via a permanent link that will never expire or change. As such, datasets/models with DOIs are intended to persist in perpetuity, and may only be deleted upon filing a request with our support.

## How are DOIs being assigned by Hugging Face? 

We have partnered with [DataCite](https://datacite.org) to allow registered Hub users to request a DOI for their model or dataset. Once they’ve filled out the necessary metadata, they receive a shiny new DOI 🌟!

<kbd>
  <img alt="Cite DOI" src="assets/107_launching_doi/cite-modal.jpeg">
</kbd>

If ever there’s a new version of a model or dataset, the DOI can easily be updated, and the previous version of the DOI gets outdated. This makes it easy to refer to a specific version of an object, even if it has changed since.


Have ideas for more improvements we can make? Many features just like these come directly from community feedback. Drop us a note or tweet us at [@HuggingFace](https://twitter.com/huggingface) to share yours or open an issue on [huggingface/hub-docs](https://github.com/huggingface/hub-docs/issues) 🤗
