---
title: "The Hugging Face Hub for Galleries, Libraries, Archives and Museums"
thumbnail: /blog/assets/144_hf_hub_glam_guide/thumbnail.png
authors:
- user: davanstrien
---

## The Hugging Face Hub for Galleries, Libraries, Archives and Museums 



### What is the Hugging Face Hub?

Hugging Face aims to make high-quality machine learning accessible to everyone. This goal is pursued in various ways, including developing open-source code libraries such as the widely-used Transformers library, offering [free courses](https://huggingface.co/learn), and providing the Hugging Face Hub. 

The [Hugging Face Hub](https://huggingface.co/) is a central repository where people can share and access machine learning models, datasets and demos. The Hub hosts over 190,000 machine learning models, 33,000 datasets and over 100,000 machine learning applications and demos. These models cover a wide range of tasks from pre-trained language models, text, image and audio classification models, object detection models, and a wide range of generative models. 

The models, datasets and demos hosted on the Hub span a wide range of domains and languages, with regular community efforts to expand the scope of what is available via the Hub. This blog post intends to offer people working in or with the galleries, libraries, archives and museums (GLAM) sector to understand how they can use &mdash; and contribute to &mdash; the Hugging Face Hub.

You can read the whole post or jump to the most relevant sections! 

- If you don't know what the Hub is, start with: [What is the Hugging Face Hub?](#what-is-the-hugging-face-hub)
- If you want to know how you can find machine learning models on the Hub, start with: [How can you use the Hugging Face Hub: finding relevant models on the Hub](#how-can-you-use-the-hugging-face-hub-finding-relevant-models-on-the-hub)
- If you want to know how you can share GLAM datasets on the Hub, start with [Walkthrough: Adding a GLAM dataset to the Hub?](#walkthrough-adding-a-glam-dataset-to-the-hub)
- If you want to see some examples, check out: [Example uses of the Hugging Face Hub](#example-uses-of-the-hugging-face-hub)

## What can you find on the Hugging Face Hub?

### Models 

The Hugging Face Hub provides access to machine learning models covering various tasks and domains. Many machine learning libraries have integrations with the Hugging Face Hub, allowing you to directly use or share models to the Hub via these libraries.

### Datasets
The Hugging Face hub hosts over 30,000 datasets. These datasets cover a range of domains and modalities, including text, image, audio and multi-modal datasets. These datasets are valuable for training and evaluating machine learning models.

### Spaces

Hugging Face [Spaces](https://huggingface.co/docs/hub/spaces) is a platform that allows you to host machine learning demos and applications. These Spaces range from simple demos allowing you to explore the predictions made by a machine learning model to more involved applications. 

Spaces make hosting and making your application accessible for others to use much more straightforward. You can use Spaces to host [Gradio](gradio.app/) and [Streamlit](https://streamlit.io/) applications, or you can use Spaces to [custom docker images](https://huggingface.co/docs/hub/spaces-sdks-docker). Using Gradio and Spaces in combination often means you can have an application created and hosted with access for others to use within minutes. You can use Spaces to host a Docker image if you want complete control over your application. There are also Docker templates that can give you quick access to a hosted version of many popular tools, including the [Argailla](https://argilla.io/) and [Label Studio](https://labelstud.io/) annotations tools.

## How can you use the Hugging Face Hub: finding relevant models on the Hub

There are many potential use cases in the GLAM sector where machine learning models can be helpful. Whilst some institutions may have the resources required to train machine learning models from scratch, you can use the Hub to find openly shared models that either already do what you want or are very close to your goal.

As an example, if you are working with a collection of digitized Norwegian documents with minimal metadata. One way to better understand what's in the collection is to use a Named Entity Recognition (NER) model. This model extracts entities from a text, for example, identifying the locations mentioned in a text. Knowing which entities are contained in a text can be a valuable way of better understanding what a document is about.

We can find NER models on the Hub by filtering models by task. In this case, we choose `token-classification`, which is the task which includes named entity recognition models. [This filter](https://huggingface.co/datasets?task_categories=task_categories:token-classification) returns models labelled as doing `token-classification`. Since we are working with Norwegian documents, we may also want to [filter by language](https://huggingface.co/models?pipeline_tag=token-classification&language=no&sort=downloads); this gets us to a smaller set of models we want to explore. Many of these models will also contain a [model widget](https://huggingface.co/saattrupdan/nbailab-base-ner-scandi), allowing us to test the model. 


![](https://i.imgur.com/9V9xni5.png)

A model widget can quickly show how well a model will likely perform on our data. Once you've found a model that interests you, the Hub provides different ways of using that tool. If you are already familiar with the Transformers library, you can click the use in Transformers button to get a pop-up which shows how to load the model in Transformers.

![](https://i.imgur.com/E9MiMi9.png)


![](image/media/image4.png)

If you prefer to use a model via an API, clicking the
`deploy` button in a model repository gives you various options for hosting the model behind an API. This can be particularly useful if you want to try out a model on a larger amount of data but need the infrastructure to run models locally.

A similar approach can also be used to find relevant models and datasets
on the Hugging Face Hub.

## Walkthrough: how can you add a GLAM dataset to the Hub?

We can make datasets available via the Hugging Face hub in various ways. I'll walk through an example of adding a CSV dataset to the Hugging Face hub. 

<figure class="image table text-center m-0 w-full">
    <video 
        alt="Uploading a file to the Hugging Face Hub"
        style="max-width: 70%; margin: auto;"
        autoplay loop autobuffer muted playsinline
    >
      <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/Upload%20dataset%20to%20hub.mp4" type="video/mp4">
  </video>
</figure>

*Overview of the process of uploading a dataset to the Hub via the browser interface*

For our example, we'll work on making the [On the Books Training Set
](https://cdr.lib.unc.edu/concern/data_sets/6q182v788?locale=en) available via the Hub. This dataset comprises a CSV file containing data that can be used to train a text classification model. Since the CSV format is one of the [supported formats](https://huggingface.co/docs/datasets/upload_dataset#upload-dataset) for uploading data to the Hugging Face Hub, we can share this dataset directly on the Hub without needing to write any code. 

### Create a new dataset repository 

The first step to uploading a dataset to the Hub is to create a new dataset repository. This can be done by clicking the `New Dataset` button on the dropdown menu in the top right-hand corner of the Hugging Face hub.

![Creating a new dataset repository](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/create_dataset.png)

Once you have done this you can choose a name for your new dataset repository. You can also create the dataset under a different owner i.e. an organization, and optionally specify a license. 

![Choosing a name](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/new_repository.png)


### Upload files

Once you have created a dataset repository you will need to upload the data files. You can do this by clicking on `Add file` under the `Files` tab on the dataset repository. 

![Add files button](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/add-files.png)

You can now select the data you wish to upload to the Hub. 

![Adding files to the Hub](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/file-upload.png)

You can upload a single file or multiple files using the upload interface. Once you have uploaded your file, you commit your changes to finalize the upload. 

![Commit your new files](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/commit.png)


### Adding metadata

It is important to add metadata to your dataset repository to make your dataset more discoverable and helpful for others. This will allow others to find your dataset and understand what it contains. 

You can edit metadata using the `Metadata UI` editor. This allows you to specify the license, language, tags etc., for the dataset. 

![Example metadata](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/add-metadata.png)

It is also very helpful to outline in more detail what your dataset is, how and why it was constructed, and it's strengths and weaknesses. This can be done in a dataset repository by filling out the `README.md` file. This file will serve as a [dataset card](https://huggingface.co/docs/datasets/dataset_card) for your dataset. A dataset card is a semi-structured form of documentation for machine learning datasets that aims to ensure datasets are sufficiently well documented. When you edit the `README.md` file you will be given the option to import a template dataset card. This template will give you helpful prompts for what is useful to include in a dataset card. 

*Tip: Writing a good dataset card can be a lot of work. However, you do not need to do all of this work in one go necessarily, and because people can ask questions or make suggestions for datasets hosted on the Hub the processes of documenting datasets can be a collective activity.*

### Datasets preview

Once we've uploaded our dataset to the Hub, we'll get a preview of the dataset. The dataset preview can be a beneficial way of better understanding the dataset. 

![Dataset server preview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-hub-glam-guide/server-view.png)

### Other ways of sharing datasets

You can use many other approaches for sharing datasets on the Hub. The datasets [documentation](https://huggingface.co/docs/datasets/share) will help you better understand what will likely work best for your particular use case. 

## Why might Galleries, Libraries, Archives and Museums want to use the Hugging Face hub?

There are many different reasons why institutions want to contribute to
the Hugging Face Hub:

- **Exposure to a new audience**: the Hub has become a central destination for people working in machine learning, AI and related fields. Sharing on the Hub will help expose your collections and work to this audience. This also opens up the opportunity for further collaboration with this audience.

- **Community:** The Hub has many community-oriented features, allowing users and potential users of your material to ask questions and engage with materials you share via the Hub. Sharing trained models and machine learning datasets also allows people to build on each other's work and lowers the barrier to using machine learning in the sector.

- **Diversity of training data:** One of the barriers to the GLAM using machine learning is the availability of relevant data for training and evaluation of machine learning models. Machine learning models that work well on benchmark datasets may not work as well on GLAM organizations' data. Building a community to share domain-specific datasets will ensure machine learning can be more effectively pursued in the GLAM sector.

- **Climate change:** Training machine learning models produces a carbon footprint. The size of this footprint depends on various factors. One way we can collectively reduce this footprint is to share trained models with the community so that people aren't duplicating the same models (and generating more carbon emissions in the process).

## Example uses of the Hugging Face Hub
Individuals and organizations already use the Hugging Face hub to share machine learning models, datasets and demos related to the GLAM sector.

### [BigLAM](https://huggingface.co/biglam)
An initiative developed out of the [BigScience project](https://bigscience.huggingface.co/) is focused on making datasets from GLAM with relevance to machine learning are made more accessible. BigLAM has so far made over 30 datasets related to GLAM available via the Hugging Face hub.

### [Nasjonalbiblioteket AI Lab](https://huggingface.co/NbAiLab) 
The AI lab at the National Library of Norway is a very active user of the Hugging Face hub, with ~120 models, 23 datasets and six machine learning demos shared publicly. These models include language models trained on Norwegian texts from the National Library of Norway and Whisper (speech-to-text) models trained on Sámi languages.

### [Smithsonian Institution](https://huggingface.co/Smithsonian)

The Smithsonian shared an application hosted on Hugging Face Spaces, demonstrating two machine learning models trained to identify Amazon fish species. This project aims to empower communities with tools that will allow more accurate measurement of fish species numbers in the Amazon. Making tools such as this available via a Spaces demo further lowers the barrier for people wanting to use these tools.

<html>
<iframe
	src="https://smithsonian-amazonian-fish-classifier.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
</html>

[Source](https://huggingface.co/Smithsonian)


## Hub features for Galleries, Libraries, Archives and Museums

The Hub supports many features which help make machine learning more accessible. Some features which may be particularly helpful for GLAM institutions include:

- **Organizations**: you can create an organization on the Hub. This allows you to create a place to share your organization's artefacts.
- **Minting DOIs**: A [DOI](https://www.doi.org/) (Digital Object Identifier) is a persistent digital identifier for an object. DOIs have become essential for creating persistent identifiers for publications, datasets and software. A persistent identifier is often required by journals, conferences or researcher funders when referencing academic outputs. The Hugging Face Hub supports issuing DOIs for models, datasets, and demos shared on the Hub.
- **Usage tracking**: you can view download stats for datasets and models hosted in the Hub monthly or see the total number of downloads over all time. These stats can be a valuable way for institutions to demonstrate their impact.
- **Script-based dataset sharing**: if you already have dataset hosted somewhere, you can still provide access to them via the Hugging Face hub using a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script).
- **Model and dataset gating**: there are circumstances where you want more control over who is accessing models and datasets. The Hugging Face hub supports model and dataset gating, allowing you to add access controls.

## How can I get help using the Hub?

The Hub [docs](https://huggingface.co/docs/hub/index) go into more detail about the various features of the Hugging Face Hub. You can also find more information about [sharing datasets on the Hub](https://huggingface.co/docs/datasets/upload_dataset) and information about [sharing Transformers models to the Hub](https://huggingface.co/docs/transformers/model_sharing).

If you require any assistance while using the Hugging Face Hub, there are several avenues you can explore. You may seek help by utilizing the [discussion forum](https://discuss.huggingface.co/) or through a [Discord](https://discord.com/invite/hugging-face-879548962464493619). 
