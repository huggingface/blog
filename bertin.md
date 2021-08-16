---
title: "BERTIN - Perplexity Sampling for efficient pre-training Language Models in Spanish"
thumbnail: /blog/assets/25_bertin/bertin.png
---

<h1>BERTIN - Perplexity Sampling for efficient pre-training of Spanish Language Models</h1>

<div class="blog-metadata">
    <small>Published August 15, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/bertin.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/versae">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1593016943046-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>versae</code>
            <span class="fullname">Javier de la Rosa</span>
        </div>
    </a>
</div>

<div class="author-card">
    <a href="/edugp">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1596046044642-noauth.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>edugp</code>
            <span class="fullname">Eduardo González Ponferrada</span>
        </div>
    </a>
</div>

<div class="author-card">
    <a href="/paulo">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1626374472249-60d433541af6803389820846.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>paulo</code>
            <span class="fullname">Paulo Villegas</span>
        </div>
    </a>
</div>

<div class="author-card">
    <a href="/Pablogps">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1624531995423-60d42257cf72c631f97af574.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>Pablogps</code>
            <span class="fullname">Pablo González de Prado</span>
        </div>
    </a>
</div>

<div class="author-card">
    <a href="/mrm8488">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1622844686495-5e4318d616b09a31220980d6.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>mrm8488</code>
            <span class="fullname">Manuel Romero</span>
        </div>
    </a>
</div>

<div class="author-card">
    <a href="/mariagrandury">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1625139502147-5f9c00a5777efc07d7f1e4be.png?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>mariagrandury</code>
            <span class="fullname">María Grandury</span>
        </div>
    </a>
</div>

## How to train state-of-the-art language models on a single TPU in a week

BERTIN is a project aimed at pre-training Spanish RoBERTa-based models from scratch efficiently, under limited compute resources.  
To achieve this goal, we explored different perplexity-based data sampling techniques to extract sub-samples of a large corpus targeting an increased prevalence of high-quality documents.  
The BERTIN project was part of the [Flax/JAX Community Event](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104), organized by [HuggigFace](https://huggingface.co) in which Google Cloud provided free TPUv3-8 for a week. During this time, we managed to pre-train several RoBERTa models in Spanish, achieving results comparable to models trained on massive private datasets using supercomputers.  
