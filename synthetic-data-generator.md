---
title: "Introducing the Synthetic Data Generator - build Datasets with Natural Language"
thumbnail: /blog/assets/synthetic-data-generator/thumbnail.png
authors:
- user: davidberenstein1957
---

# Synthetic Data Generator - a No-code Approach to Building Datasets with Natural Language

Introducing the Synthetic Data Generator, a user-friendly application that takes a no-code approach to creating custom datasets with Large Language Models (LLMs). The best part: A simple step-by-step process, making dataset creation a non-technical breeze, allowing anyone to create datasets and models in minutes and without any code.

<details>
  <summary>What is synthetic data and why is it useful?</summary>
  <p>
    Synthetic data is artificially generated information that mimics real-world data. It allows overcoming data limitations by expanding or enhancing datasets.
  </p>
</details>

## From Prompt to Model

The synthetic data generator takes your custom prompt and returns a dataset for your use case, using a synthetic data pipeline. In the background, this is powered by [distilabel](https://distilabel.argilla.io/latest/) and the [free Hugging Face text-generation API](https://huggingface.co/docs/api-inference/en/index) but we don’t need to worry about these complexities and we can focus on using the UI.

### Supported Tasks

At the time of writing, there are two supported tasks, text classification and supervised fine-tuning (SFT). However, we will be adding tasks like evaluation and RAG over time based on demand.

#### Text Classification

Text classification is a common task to sort documents like customer reviews, social media posts, or news articles. The task to generate a dataset relies on two different steps that use LLMs. We first generate diverse texts and then add labels to them. A good example of a generated dataset for text classification is the [argilla/synthetic-text-classification-news](https://huggingface.co/datasets/argilla/synthetic-text-classification-news), which classifies synthetic news articles into 8 different classes.

<iframe
  src="https://huggingface.co/datasets/argilla/synthetic-text-classification-news/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

#### Supervised Fine-Tuning (SFT)

SFT is a generative task for interactions with LLMs in chatbot-like scenarios. This task is more complex and consists of two generation phases. We first generate input prompts and then generate responses to those prompts. A good example of a generated dataset for SFT is [argilla/synthetic-sft-customer-support-single-turn](https://huggingface.co/datasets/argilla/synthetic-sft-customer-support-single-turn), which highlights an example of an LLM that handles customer support for the synthetic data generator.

<iframe
  src="https://huggingface.co/datasets/argilla/synthetic-sft-customer-support-single-turn/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Generally, we can generate 50 and 20 samples per minute for text classification and SFT, respectively. However, you can scale this up by using your own account. We will get back to this later but let’s first dive into the UI.

### Let’s generate our first dataset

We will create a basic SFT dataset. We start with a login, which redirects us to the page where we configure the tool’s access to your user and organisations. Make sure to include the organisations for which you want to generate datasets. In case of a failed authentication, you can always [reset the connection](https://huggingface.co/settings/connected-applications).

After the login, the UI guides you through a straightforward three-step process:

#### 1. Describe Your Dataset

Start by providing a description of the dataset you want to create, including example use cases to help the generator understand your needs. Make sure to describe your dataset and don’t come up with a prompt from scratch. When you hit the “Create” button, a sample dataset will be created, and you can continue with step 2.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/synthetic-data-generator/step1.png" style="width: 100%;">

#### 2. Configure and Refine

Refine your generated sample dataset by adjusting the system prompt and task-specific settings to get the results you're after. You can iterate on these configurations by hitting the “Save” button and regenerating your sample dataset. When you are satisfied with the config, continue to step 3.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/synthetic-data-generator/step2.png" style="width: 100%;">


#### 4. Generate and Push

Fill out general information about the dataset name and organisation. Additionally, you can define the number of samples to generate and the temperature to use for the generation. This temperature represents the creativity of the generations. Now, let’s hit the “Generate” button to start a full generation. The output will be saved directly to Argilla and the Hugging Face Hub.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/synthetic-data-generator/step3.png" style="width: 100%;">

We can now press the “Open in Argilla” button and directly dive into our generated dataset.

### Reviewing the Dataset

Even when dealing with synthetic data, it is important to understand and look at your data, which is why we created a direct integration with [Argilla](https://argilla.io/), a collaboration tool for AI engineers and domain experts to build high-quality datasets. This allows you to effectively explore and evaluate the synthetic dataset through powerful features like semantic search and composable filters. You can learn more about them in [our guide](https://docs.argilla.io/latest/how_to_guides/annotate/). Afterwards, we can export the curated dataset to the Hugging Face Hub, and continue to fine-tune our model.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/synthetic-data-generator/argilla.png" style="width: 100%;">

### Training a Model

Don’t worry, even creating powerful AI models can be done without code nowadays using [AutoTrain](https://huggingface.co/autotrain). To better understand how it works, you can take a look at its [documentation](https://huggingface.co/docs/autotrain/en/index). For now, we will simply [create our own AutoTrain deployment](https://huggingface.co/spaces/autotrain-projects/autotrain-advanced?duplicate=true) and log in as we’ve done before for the synthetic data generator.

Remember the [argilla/synthetic-text-classification-news dataset](https://huggingface.co/datasets/argilla/synthetic-text-classification-news) from the beginning? Let’s train a model that can correctly classify these examples. We just need to select the task “Text Classification” and provide the correct “Dataset source”. Then, choose a nice project name and press play! The pop-up that warns about costs can be ignored because we are still working on the free Hugging Face offering.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/synthetic-data-generator/autotrain.png" style="width: 100%;">

Et voilà, after a couple of minutes, we’ve got [our very own model](https://huggingface.co/argilla/synthetic-text-classification-news-autotrain-model)! All that remains is to [deploy it as a live service](https://ui.endpoints.huggingface.co/davidberenstein1957/new?repository=argilla%2Fsynthetic-text-classification-news-autotrain-model) or to [use it as a text-classification pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextClassificationPipeline) with some minimal Python code.

## Additional Features

Even though you can go from prompts to dedicated models without knowing anything about coding, some people might like the option to customize and scale their deployment with some additional technical features.

### Improving Speed and Accuracy

You can improve speed and accuracy by using your own API keys and by using different parameters or models. First, you must [duplicate the synthetic data generator](https://huggingface.co/spaces/argilla/synthetic-data-generator?duplicate=true). Make sure you create is as a private Space to ensure nobody else can access it. Next, you can [change some environment variables](https://github.com/argilla-io/synthetic-data-generator?tab=readme-ov-file#environment-variables). For instance, let’s increase our processing speed by setting `DEFAULT_BATCH_SIZE` to `20`, use OpenAI by setting `BASE_URL` to `https://api.openai.com/v1/`, `MODEL` to `gpt-4o` and lastly, `API_KEY` to [a valid OpenAI API key](https://platform.openai.com/api-keys). Don’t forget to pass your `HF_TOKEN` to still be able to push datasets to the Hub.

### Local Deployment

Besides hosting the tool on Hugging Face Spaces, we also offer it as an open-source tool under an Apache 2 license, which means you can go [to GitHub](https://github.com/argilla-io/synthetic-data-generator) and use, modify, and adapt it however you need. You can [install it as a Python package](https://github.com/argilla-io/synthetic-data-generator?tab=readme-ov-file#installation) through a simple `pip install synthetic-dataset-generator`.

### Customising Pipelines

Each synthetic data pipeline is based on [distilabel](https://distilabel.argilla.io/latest/), the framework for synthetic data and AI feedback. distilabel is open source, and the cool thing about the pipeline code is that it is sharable and reproducible. You can, for example, [find the pipeline for the argilla/synthetic-text-classification-news dataset](https://huggingface.co/datasets/argilla/synthetic-text-classification-news/blob/main/pipeline.py) within the repository on the Hub. Alternatively, you can find many [other distilabel datasets along with their pipelines](https://huggingface.co/datasets?other=distilabel).

## What’s Next?

The Synthetic Data Generator already offers a lot of cool features to make it useful for any data or model lover but we have some interesting [directions for improvements on our GitHub](https://github.com/argilla-io/synthetic-data-generator/issues) and we invite you to contribute too! Some things we are working on are:

- Retrieval Augmented Generation (RAG)
- Evaluation with LLMs as a Judge

[Start synthesizing](https://huggingface.co/spaces/argilla/synthetic-data-generator)
