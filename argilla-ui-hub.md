---
title: "Argilla 2.4: Easily Build Fine-Tuning and Evaluation Datasets on the Hub — No Code Required"
thumbnail: /blog/assets/argilla-ui-hub/thumbnail.png
authors:
- user: nataliaElv
- user: burtenshaw
- user: dvilasuero
---

# Argilla 2.4: Easily Build Fine-Tuning and Evaluation Datasets on the Hub — No Code Required

We are incredibly excited to share the most impactful feature since Argilla joined Hugging Face: you can prepare your AI datasets without any code, getting started from any Hub dataset! Using Argilla’s UI, you can easily import a dataset from the Hugging Face Hub, define questions, and start collecting human feedback.

> [!NOTE]
> Not familiar with Argilla? Argilla is a free, open-source data-centric tool. Using Argilla, AI developers and domain experts can collaborate and build high-quality datasets. Argilla is part of the Hugging Face family and fully integrated with the Hub. Want to know more? Here’s an [intro blog post](https://huggingface.co/blog/dvilasuero/argilla-2-0).

Why is this new feature important to you and the community?

- The Hugging Face hub contains 230k datasets you can use as a foundation for your AI project.
- It simplifies collecting human feedback from the Hugging Face community or specialized teams.
- It democratizes dataset creation for users with extensive knowledge about a specific domain who are unsure about writing code.

## Use cases

This new feature democratizes building high-quality datasets on the Hub:

- If you have published an open dataset and want the community to contribute, import it into a public Argilla Space and share the URL with the world!
- If you want to start annotating a new dataset from scratch, upload a CSV to the Hub, import it into your Argilla Space, and start labeling!
- If you want to curate an existing Hub dataset for fine-tuning or evaluating your model,  import the dataset into an Argilla Space and start curating!
- If you want to improve an existing Hub dataset to benefit the community, import it into an Argilla Space and start giving feedback!


## How it works

First, you need to deploy Argilla. The recommended way is to deploy on Spaces [following this guide](https://docs.argilla.io/latest/getting_started/quickstart/). The default deployment comes with Hugging Face OAuth enabled, meaning your Space will be open for annotation contributions from any Hub user. OAuth is perfect for use cases when you want the community to contribute to your dataset. If you want to restrict annotation to you and other collaborators, [check this guide](https://docs.argilla.io/latest/getting_started/how-to-configure-argilla-on-huggingface/) for additional configuration options.

<video controls autoplay title="Import a HF dataset from the Hub in the Argilla UI">
<source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-ui-hub/import_hub_dataset.mp4" type="video/mp4">
Import a HF dataset from the Hub in the Argilla UI
</video>

Once Argilla is running, sign in and click the “Import dataset from Hugging Face” button on the Home page. You can start with one of our example datasets or input the repo id of the dataset you want to use.
> [!NOTE]
> In this first version, the Hub dataset must be public. If you are interested in support for private datasets, we’d love to hear from you on [GitHub](https://github.com/argilla-io/argilla/issues/new/choose).

Argilla automatically suggests an initial configuration based on the dataset’s features, so you don’t need to start from scratch, but you can add questions or remove unnecessary fields. Fields should include the data you want feedback on, like text, chats, or images. Questions are the feedback you wish to collect, like labels, ratings, rankings, or text. All changes are shown in real time, so you can get a clear idea of the Argilla dataset you’re configuring. 

Once you’re happy with the result, click “Create dataset” to import the dataset with your configuration. Now you’re ready to give feedback!

You can try this for yourself by following the [quickstart guide](https://docs.argilla.io/latest/getting_started/quickstart/). It takes under 5 minutes!

This new workflow streamlines the import of datasets from the Hub, but you can still [import datasets using Argilla’s Python SDK](https://docs.argilla.io/latest/how_to_guides/dataset/) if you need further customization.

We’d love to hear your thoughts and first experiences. Let us know on [GitHub](https://github.com/argilla-io/argilla/issues/new/choose) or the [HF Discord](https://discord.com/invite/hugging-face-879548962464493619)!
