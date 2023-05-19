---
title: "The Hugging Face Hub for Galleries, Libraries, Archives and Museums"
thumbnail: TODO
authors:
- user: davanstrien
  guest: false
---

## The Hugging Face Hub for Galleries, Libraries, Archives and Museums 


<!-- {blog_metadata} -->
<!-- {authors} -->

Hugging Face aims to make high-quality machine learning accessible to everyone. They achieve this in various ways, including developing open-source code libraries such as the widely-used Transformers library, offering a free course, and providing the Hugging Face Hub. 

## What is the Hugging Face Hub?

The Hugging Face Hub is a central repository where people can share and access machine learning models, datasets and demos. The Hub hosts over 198,000 machine learning models, 33,000 datasets and over 100,000 machine learning applications and demos. These models cover a wide range of tasks from pre-trained language models, text, image and audio classification models, and object detection models, as well as a wide range of generative models. The models, datasets and demos hosted on the Hub span a wide range of domains and languages, with regular community efforts to expand the scope of what is available via the Hub.  This blog post is intended to offer people working in or with the galleries, libraries, archives and museums sector to understand how they can make use of --- and contribute to --- the Hugging Face Hub.

## What can you find on the Hugging Face Hub?

### Models 

The Hugging Face Hub provides access to machine learning models covering a wide range of tasks and domains. A large number of machine learning libraries have integrations with the Hugging Face Hub allowing you to directly use or share models to the hub via these libraries.

### Datasets
The Hugging Face hub hosts over 30,000 datasets. These datasets cover a range of domains and modalities including text, image, audio and multi-modal datasets. These datasets are valuable for training and evaluating machine learning models.

### Spaces

Hugging Face Spaces is a platform that allows you to host machine learning demos and applications. These range from simple demos allowing you to explore the predictions made by a machine learning model to more involved applications. Spaces makes the process of hosting and making available your application accessible for others to use much easier. Spaces can be used to host Gradio and Streamlit applications or can be used to host custom docker images. Using Gradio and Spaces in combination often means you can have an application created and hosted with access for others to use within minutes. If you want full control over your application you can use Spaces to host a Docker image. There are also Docker templates that can give you quick access to a hosted version of many popular tools including the Argilla and Label Studio annotations tools.

### How can you use the Hugging Face Hub: finding relevant models on the Hub

There are many potential use cases in the GLAM sector where machine learning models can be helpful. Whilst some institutions may have the resources required to train machine learning models from scratch, you can use the Hub to find openly shared models which either already do what you want or are very close to what you are trying to achieve.

Let's say you are working with a collection of digitised Norwegian documents with minimal metadata. One way of getting a better understanding of what\'s in the collection could be to use a Named Entity Recognition (NER) model. This model extracts entities from a text, for example extracting all the locations mentioned in a text. Knowing which entities are contained in a text can be a valuable way of better understanding what a document is about.

We can find NER models on the hub by filtering models by task. In this case, we choose `token-classification`, which is the task which includes named entity recognition models. [This filter](https://huggingface.co/datasets?task_categories=task_categories:token-classification) returns models which are labelled as doing `token-classification`.  Since we are working with Norwegian documents, we may also want to [filter by language](https://huggingface.co/models?pipeline_tag=token-classification&language=no&sort=downloads); this gets us to a smaller set of models we want to explore. Many of these models will also contain a [model widget](https://huggingface.co/saattrupdan/nbailab-base-ner-scandi) which allows us to test out the model. 


![](https://i.imgur.com/9V9xni5.png)

A model widget can quickly give us a sense of how well a model will likely perform on our data. Once you've found a model that interests you, the Hub provides different ways of using that tool. If you are already familiar with the Transformers library, you can click the use in Transformers button to get a pop-up which shows how to load the model in Transformers.

![](https://i.imgur.com/E9MiMi9.png)


![](image/media/image4.png)

You may instead prefer to use the model via an API. We can click on the
`deploy` button to get various options for deploying a model. This can
be particularly useful if you want to try out a model on a larger amount
of data but don't have the infrastructure to run models locally.


A similar approach can also be used to find relevant models and datasets
on the Hugging Face Hub.

## Contributing to the Hub

There are various ways in which GLAM institutions can contribute to the
Hugging Face Hub, including by sharing models, datasets and demos.

### Why might Galleries, Libraries, Archives and Museums want to use the Hugging Face hub?

There are many different reasons why institutions want to contribute to
the Hugging Face Hub:

- **Exposure to a new audience**: the Hub has become a central  destination for people working in machine learning, AI and related fields. Sharing on the Hub will help expose your collections and work to this audience. This also opens up the opportunity for further collaboration with this audience.

- **Community:** the hub has many community-oriented features, allowing users and potential users of your material to ask questions and engage with materials you share via the Hub. Sharing trained models and machine learning datasets also allows people to build on each other's work and lowers the barrier to using machine learning in the sector.

- **Diversity of training data:** one of the barriers to the GLAM using machine learning is the availability of relevant data for training and evaluation of machine learning models. Machine learning models which work well on benchmark datasets used in machine learning may not translate to the types of data used by GLAM organizations. Building a community to share domain-specific datasets will ensure machine learning can be more effectively pursued in the GLAM sector.

- **Climate change:** training machine learning models results in a carbon footprint. The size of this footprint depends on various factors, but one way we can collectively reduce this footprint is to share trained models with the community so that people aren't duplicating the same models (and generating more carbon emissions in the process).

### Example uses of the Hugging Face Hub
There are already individuals and organizations using the Hugging Face hub to share machine learning models, datasets and demos related to the GLAM sector.

#### BigLAM 
[BigLAM](https://huggingface.co/biglam), an initiative developed out of the BigScience project, is focused on making datasets from GLAM with relevance to machine learning are made more accessible. BigLAM has so far made over 30 datasets related to GLAM available via the Hugging Face hub.

#### [Nasjonalbiblioteket AI Lab](https://huggingface.co/NbAiLab) 
The AI lab at the National Library of Norway is a very active user of the Hugging Face hub, with ~120 models, 23 datasets and six machine learning demos shared publicly. These models include language models trained on Norwegian texts from the National Library of Norway and Whisper (speech-to-text) models trained on Sámi languages.

#### [Smithsonian Institution](https://huggingface.co/Smithsonian)

The Smithsonian has shared an application hosted on Hugging Face Spaces, which demonstrates two machine learning models trained to identify Amazon fish species. The goal of this project is to empower communities with tools that will allow for more accurate measurement of fish species numbers in the Amazon. Making tools such as this available via a Spaces demo further lowers the barrier for people wanting to use these tools.

<html>
<iframe
	src="https://smithsonian-amazonian-fish-classifier.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
</html>

[Source](https://huggingface.co/Smithsonian)


### Hub features for GLAM.

The Hub supports many features which help make machine learning more accessible. Some features which may be particularly helpful for GLAM institutions include:

- **Organizations**: you can create an organization on the Hub. This allows you to create a space under which you can share artefacts from your organizations.
- **Minting DOIs**: A [DOI](https://www.doi.org/) (Digital Object Identifier) is a persistent digital identifier for an object. DOIs have become an essential tool for creating persistent identifiers for publications, datasets and software. A persistent identifier is often required by journals, conferences or researcher funders when referencing academic outputs. The Hugging Face Hub supports issuing DOIs for models, datasets, and demos shared on the hub.
- **Usage tracking**: you can view download stats for datasets and models hosted in the hub each month or see the total number of downloads over all time. This can be a valuable way for institutions to demonstrate their impact.
- **Script-based dataset sharing**: if you already have dataset hosted somewhere, you can still provide access to them via the Hugging Face hub using a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script).
- **Model and dataset gating**: there are circumstances where you want more control over who is accessing models and datasets. The Hugging Face hub has support for model and dataset gating, which allows you to add access controls.

### How can I get help using the hub?

The hub [docs](https://huggingface.co/docs/hub/index) go into more detail about the various features of the Hugging Face Hub. You can also find more information about [sharing datasets on the Hub](https://huggingface.co/docs/datasets/upload_dataset) and information about [sharing Transformers models to the Hub](https://huggingface.co/docs/transformers/model_sharing).

If you require any assistance while using the Hugging Face Hub, there are several avenues you can explore. You may seek help by utilizing the [discussion forum](https://discuss.huggingface.co/) or through a [Discord](https://discord.com/invite/hugging-face-879548962464493619). If you have any queries that are specific to GLAM uses of the Hub, you can get in touch with Hugging Face's Machine Learning Librarian for additional guidance.
