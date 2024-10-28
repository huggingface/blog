---
title: "Argilla 2.4: Curate Hub Datasets with Human Feedback—No Code Needed"
thumbnail: /blog/assets/argilla-ui-hub/thumbnail.png
authors:
- user: nataliaElv
- user: burtenshaw
- user: dvilasuero
---

# Argilla 2.4: Curate Hub Datasets with Human Feedback—No Code Needed

We are extremely excited to share a major new feature: you can now start off your AI dataset projects without code. Using Argilla’s UI, you can easily import any dataset from the Hugging Face Hub, define questions, and start collecting human feedback.

> [!NOTE]
> Not familiar with Argilla? Argilla is a free, open-source data curation tool. Using Argilla, AI developers and domain experts can collaborate and build high quality datasets. Argilla is part of the Hugging Face family and it’s fully integrated with the Hub: from deploying it on Spaces to fine-tuning models with AutoTrain and TRL. Want to know more? Here’s an [intro blog post](https://huggingface.co/blog/dvilasuero/argilla-2-0).

Why is this new feature important to you and the community?

- There are 230k datasets on the Hugging Face hub that you can use as a foundation to your own AI project.
- For dataset publishers, it simplifies the process of collecting human feedback from the whole Hugging Face community or specialized teams and communities.
- This makes the step of going from a dataset in the Hub to human feedback workflows easier and faster. 
- This empowers users that aren’t familiar with Python or code, allowing them to create datasets using a UI instead of code.

## Use cases

This new feature enables a new set of use cases for building high quality datasets on the Hub:

- You have just published an open dataset and want the community to contribute: import it into a public Argilla Space and share the URL with the world!
- You want to start annotating a new dataset from scratch: upload a CSV to the Hub, import it into an Argilla Space and start labeling!
- You want to curate an existing Hub dataset for your use case or contribute to improve it: import it into an Argilla Space and start curating!

## How it works

First, you need to deploy Argilla. The recommended way is to deploy on Spaces, [following this guide](https://docs.argilla.io/latest/getting_started/quickstart/). The default deployment comes with Hugging Face OAuth enabled, meaning your Space will be open to any Hub users to annotate your dataset. This is perfect for use cases when you want the community to contribute to your dataset. If you want to keep the annotation restricted to you and other collaborators, [check this other guide](https://docs.argilla.io/latest/getting_started/how-to-configure-argilla-on-huggingface/).

[screen recording]

Once Argilla is running, sign in and click the “Import from Hub” button in the Home page. You can get started with one of our example datasets or input the repo id of the dataset you want to use. Argilla will automatically suggest an initial configuration based on the dataset’s features, so you don’t need to start from scratch.

> [!NOTE]
> In this first version, the Hub dataset must be public, if you are interested in support for private datasets, we’d love to hear from you on [GitHub](https://github.com/argilla-io/argilla).

The dataset’s columns will be mapped to fields and questions in Argilla. Fields include the data that you want feedback on, like text, chats, or images. Questions are the feedback you want to collect, like labels, ratings, rankings, or text. If you need, you can add and configure questions or remove unnecessary fields. All of the changes that you make will be previewed in real time, so you can see how your changes affect the dataset.

Once you’re happy with the result, you can click on “Create dataset” to import the dataset. Now you’re ready to give feedback!

This new workflow streamlines the import of datasets from the Hub, but you can still [import datasets using Argilla’s Python SDK](https://docs.argilla.io/latest/how_to_guides/dataset/) if you need further customization.

We’d love to know what you think or how your experience was, so we can make the import of Hub datasets even better. Let us know on GitHub or the HF Discord!
