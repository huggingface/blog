---
title: "[Presidio] Experimenting with Automatic PII Detection on the Hub" 
thumbnail: /blog/assets/presidio-pii-detection/presidio_report.jpg
authors:
- user: lhoestq
---

# [Presidio] Experimenting with Automatic PII Detection on the Hub

At Hugging Face we've seen the emergence of ML datasets containing Personal Identifiable Information (PII) that pose challenges to ML practitioners.

First, there are datasets with annotated PII data that are used train PII Detection Models to detect and mask PII.
They can help with online content moderation or provide anonymized databases for example.

There also exist pre-training datasets which also contain PII.
They are large scale datasets of terabytes of text typically obtained by filtering web crawls.
These datasets are generally filtered to remove certains types of PII, but small amounts can still remain.
This is due to the scale of those datasets and because PII Detection Models are generally not perfect.

To help ML practitioners with the challenges related to PII and for transparency, we are experimenting with a new feature on the Dataset Hub that uses [Presidio](https://github.com/microsoft/presidio) to show a report with an estimation of the presence of PII to users.

Presidio is an open source state-of-the-art PII detection tool that relies on detection patterns and machine learning models.

You can see an example on this [pre-training dataset](https://huggingface.co/datasets/allenai/c4), where Presidio detects small amounts of email and sensitive PII:

![presidio report](assets/presidio-pii-detection/presidio_report.jpg)

This information can be used by ML practitioners to take informed decisions before training a model, for example if they need to further filter the dataset (e.g. using Presidio).
Datasets owners can also use this information to validate their PII filtering before releasing a dataset.
