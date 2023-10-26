---
title: "Empowering Open Source Machine Learning through Dataset Sharing on the Hugging Face Hub"
thumbnail: /blog/assets/researcher-dataset-sharing/thumbnail.png
authors:
- user: davanstrien
---

## Introduction: who is this blog post for?

Are you a researcher doing data-intensive research or using machine learning as a research tool? As part of this research, you have likely created datasets for training and evaluating machine learning models, and like many researchers, you may be sharing these datasets via Google Drive, OneDrive, or your own personal server. In this post, we’ll outline why you might want to consider sharing these datasets on the Hugging Face Hub instead. 

This post outlines:

- Why researchers should openly share their data (feel free to skip this section if you are already convinced about this!)
- What the Hugging Face Hub offers for researchers who want to share their datasets.
- Resources for getting started with sharing your datasets on the Hugging Face Hub.

## Why share your data?

Machine learning is increasingly utilized across various disciplines, enhancing research efficiency in tackling diverse problems. Data remains crucial for training and evaluating models, especially when developing new machine-learning methods for specific tasks or domains. Large Language Models may not perform well on specialized tasks like bio-medical entity extraction, and computer vision models might struggle with classifying domain specific images.

Domain-specific datasets are vital for evaluating and training machine learning models, helping to overcome the limitations of existing models. Creating these datasets, however, is challenging, requiring significant time, resources, and domain expertise, particularly for annotating data. Maximizing the impact of this data is crucial for the benefit of both the researchers involved and their respective fields.

The Hugging Face Hub can help achieve this maximum impact. 

## What is the Hugging Face Hub?

The [Hugging Face Hub](https://huggingface.co/) has become the central hub for sharing open machine learning models, datasets and demos, hosting over 360,000 models and 70,000 datasets. The Hub enables people – including researchers – to access state-of-the-art machine learning models and datasets in a few lines of code. 

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/hub-datasets.png" alt="Screenshot of datasets ib the Hugging Face Hub"><br> 
<em>Datasets on the Hugging Face Hub.</em> 
 </p> 

## What does the Hugging Face Hub offer for data sharing?

This blog post won’t cover all of the features and benefits of hosting datasets on the Hugging Face Hub but will instead highlight some that are particularly relevant for researchers. 

### Visibility for your work 

The Hugging Face Hub has become the central Hub for people to collaborate on open machine learning. Making your datasets available via the Hugging Face Hub ensures it is visible to a wide audience of machine learning researchers. The Hub makes it possible to expose links between datasets, models and demos which makes it easier to see how people are using your datasets for training models and creating demos. 

### Tools for exploring and working with datasets

There are a growing number of tools being created which make it easier to understand datasets hosted on the Hugging Face Hub. 

#### Datasets Viewer 

The datasets-viewer allows people to explore and interact with datasets hosted on the Hub. This makes it much easier for people to view and explore your data without first having to download it. The datasets viewer also allows you to search and filter datasets, which can be valuable to potential dataset users, understanding the nature of a dataset more quickly. 


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/datasets-viewer.png" alt="Screenshot of a dataset viewer on the Hub showing a named entity recognition dataset"><br> 
<em>The dataset viewer for the the multiconer_v2 Named Entity Recognition dataset.</em> 
 </p> 

### Community tools 

Alongside the datasets viewer there are a growing number of community created tools for exploring datasets on the Hub.

#### Spotlight

[Spotlight](https://github.com/Renumics/spotlight) is a tool that allows you you to interactively explore datasets on the Hub with one line of code.   

<p align="center"><a href="https://github.com/Renumics/spotlight"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/scalable-data-inspection/speech_commands_vis_s.gif" width="100%"/></a></p> 

You can learn more about how you can use this tool in this [blog post](https://huggingface.co/blog/scalable-data-inspection).

#### Lilac

[Lilac](https://lilacml.com/) is a tool that aims to help you "curate better data for LLMs" and allows you to explore natural language datasets more easily. The tool allows you to semantically search your dataset (search by meaning), cluster data and gain high-level insights into your dataset.

<div style="text-align: center;">
    <iframe
        src="https://lilacai-lilac.hf.space"
        frameborder="0"
        width="850"
        height="450"
    ></iframe>
    <em>A Spaces demo of the lilac tool.</em> 
</div>

You can explore the Lilac tool further in a [demo](https://lilacai-lilac.hf.space/).

### Support for large datasets

The Hub can host large datasets; the Hub currently hosts datasets which are multiple TBs.The datasets library, which users can use to download and process datasets from the Hub, supports streaming for large datasets, making it possible to work with large datasets without downloading the entire dataset upfront. This can be invaluable for allowing researchers with less computational resources to work with your datasets. 


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/filesize.png" alt="Screenshot of the file size information for a dataset"><br> 
<em>The Hugging Face Hub can host the large datasets often created for machine learning research.</em> 
 </p> 


## API and client library interaction with the Hub

Interacting with the Hugging Face Hub via an API or a Python library is possible. This includes creating new repositories, uploading data programmatically and creating and modifying metadata for datasets. This can be powerful for research workflows where new data or annotations continue to be created. The client library also makes uploading large datasets much more accessible. 

## Community 

The Hugging Face Hub is already home to a large community of researchers, developers, artists, and others interested in using and contributing to an ecosystem of open-source machine learning. Making your datasets accessible to this community increases its visibility, opens it up to new types of users and places your datasets within the context of a larger ecosystem of models, datasets and libraries.

The Hub also has features which allow communities to collaborate more easily. This includes a discussion page for each dataset, model and Space hosted on the Hub. This means users of your datasets can quickly ask questions and discuss ideas for working with a dataset. 


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/discussion.png" alt="Screenshot of a discussion for a dataset on the Hub."><br> 
<em>The Hub makes it easy to ask questions and discuss datasets.</em> 
 </p> 


### Other important features for researchers

Some other features of the Hub may be of particular interest to researchers wanting to share their machine learning datasets on the Hub:

- Organizations allow you to collaborate with other people and share models, datasets and demos under a single organisation. This can be an excellent way of highlighting the work of a particular research project or institute. 
- Dataset loading scripts give you control over how your dataset is downloaded and processed. This makes it possible to share multiple dataset configurations and allows end users to start working with your data quickly. 
- Gated repositories allow you to add some access restrictions to accessing your dataset. 
- Download metrics are available for datasets on the Hub; this can be useful for communicating the impact of your researchers to funders and hiring committees. 
- DOIs: it’s possible to register a persistent identifier for your dataset. 

### How can I share my dataset on the Hugging Face Hub? 
TODO Links to existing docs

If you want any further help uploading a dataset to the Hub or want to upload a particularly large dataset, please contact datasets@huggingface.co. 
