---
title: "Improving Hugging Face Model Access for Kaggle Users" 
thumbnail: /blog/assets/kaggle-integration/thumbnail.png
authors:
- user: roseberryv
  guest: true
  org: kaggle
- user: megrisdal
  guest: true
  org: kaggle
- user: julien-c
- user: pcuenq
- user: reach-vb
---

Kaggle and Hugging Face users are part of one AI community. That’s why we’re excited to announce our plans to bring our platforms and communities closer to better serve AI developers everywhere.

Beginning today, Kaggle is launching an integration that enhances visibility and discoverability for Hugging Face models directly on Kaggle. 

## How to get started

You can navigate from Hugging Face models to Kaggle and vice versa. Start by visiting a Hugging Face model page like [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B). To use it in a Kaggle Notebook, you can click on “Use this model” and select “Kaggle” to open up a Kaggle notebook with a pre-populated code snippet to load the model. You can do the same from a Hugging Face model page on Kaggle by clicking the “Code” button.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kaggle-integration/new-notebook.gif" alt="Creating a new notebook using a Hugging Face model on Kaggle">

When you run a notebook on Kaggle that references a model hosted on Hugging Face Hub, we will automatically generate a Hugging Face model page if one doesn’t exist already. You don’t need to make any special changes to your code.  Additionally, when you make your notebook public, it will automatically show on the “Code” tab of the Kaggle model page.

Discover Hugging Face models and explore all the community examples in public notebooks in one place on Kaggle at [https://www.kaggle.com/models](https://www.kaggle.com/models). As more Hugging Face models are used on Kaggle, the number of models and associated code examples you can explore for inspiration will grow.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kaggle-integration/browsing-hugging-face-models-on-kaggle.gif" alt="Browsing Hugging Face models on Kaggle">

When browsing Hugging Face models on Kaggle, we want to make it easy for you to navigate back to Hugging Face to explore additional details, metadata, community usage in Hugging Face Spaces, discussion, and more. Simply click “Open in Hugging Face” on the Kaggle model page.


## How does this work with private and consent-gated Hugging Face models?

If you use a private Hugging Face model in your Kaggle notebook, authenticate via your Hugging Face account as normal (add your HF_TOKEN in the “Add-ons > Secrets” menu in the notebook editor). A Hugging Face model page won’t be generated on Kaggle. 

If you want to access a consent-gated model in your Kaggle notebook, you’ll need to request access to it using a Hugging Face account and follow the prompts on your browser's Hugging Face model page as normal. [Hugging Face has documentation](https://huggingface.co/docs/hub/en/models-gated#access-gated-models-as-a-user) to guide you through this process. Otherwise, the integration will work the same as for non-gated models.


## What’s next

We’re actively working on a solution to seamlessly use Hugging Face models in Kaggle competitions that require offline notebook submissions. While this will take a few more months to complete, we believe the wait will be worth it. 

You can read Kaggle’s position on [“AI Competitions as the gold standard for empirical rigor for GenAI evaluation”](https://huggingface.co/papers/2505.00612) to understand why it’s so important for us to get this part of the integration right! But tl;dr – Kaggle is highly sensitive to data leakage and its impact on model contamination.  Our goal is to design this integration to preserve the integrity of our competitions and their vital role in the industry, while enabling seamless access for Kaggle competitors to build with the best models from Hugging Face!

We’d love to hear your feedback in the meantime - [share your thoughts and ideas here](https://huggingface.co/spaces/kaggle/hf-integration-feedback/discussions/1)!

Happy Kaggling! 
