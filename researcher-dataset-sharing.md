---
title: "Creating open machine learning datasets? Share them on the Hugging Face Hub!"
thumbnail: /blog/assets/researcher-dataset-sharing/thumbnail.png
authors:
- user: davanstrien
- user: cfahlgren1
- user: lhoestq
---

## Who is this blog post for?

Are you a researcher doing data-intensive research or using machine learning as a research tool? As part of this research, you have likely created datasets for training and evaluating machine learning models, and like many researchers, you may be sharing these datasets via Google Drive, OneDrive, or your own personal server. In this post, we’ll outline why Hugging Face Hub is the best place on the internet to host open datasets.

Many of the world's leading research institutions and companies such as Meta, Google, OpenAI, and THUDM are using the Hugging Face Hub to host their datasets. By hosting a dataset on the Hugging Face Hub, you instantly get access to many features that can maximize the impact of your dataset:

- Generous Limits
- Dataset Viewer
- Third Party Library Support
- SQL Console
- Access Controls
- Reach and Visibility

## Why share your data?

Machine learning is increasingly utilized across various disciplines, enhancing research efficiency in tackling diverse problems. Data remains crucial for training and evaluating models, especially when developing new machine-learning methods for specific tasks or domains. Large Language Models may not perform well on specialized tasks like bio-medical entity extraction, and computer vision models might struggle with classifying domain specific images.

Domain-specific datasets are vital for evaluating and training machine learning models, helping to overcome the limitations of existing models. Creating these datasets, however, is challenging, requiring significant time, resources, and domain expertise, particularly for annotating data. Maximizing the impact of this data is crucial for the benefit of both the researchers involved and their respective fields.

The Hugging Face Hub can help achieve this maximum impact. 

## What is the Hugging Face Hub?

The [Hugging Face Hub](https://huggingface.co/) has become the central hub for sharing open machine learning models, datasets and demos, hosting over 360,000 models and 70,000 datasets. The Hub enables people – including researchers – to access state-of-the-art machine learning models and datasets in a few lines of code. 

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/hub-datasets.png" alt="Screenshot of datasets in the Hugging Face Hub"><br> 
<em>Datasets on the Hugging Face Hub.</em> 
 </p> 

## What does the Hugging Face Hub offer for data sharing?

This blog post won’t cover all of the features and benefits of hosting datasets on the Hugging Face Hub but will instead highlight some that are particularly relevant for researchers. 

### Reach and Visibility

The Hugging Face Hub has become the central Hub for people to collaborate on open machine learning. Making your datasets available via the Hugging Face Hub ensures it is visible to a wide audience of machine learning researchers. The Hub makes it possible to expose links between datasets, models, academic papers, and demos which makes it easier to see how people are using your datasets for training models and creating demos.

Each dataset on the Hub includes a customizable README file that serves as comprehensive documentation. This README allows you to provide detailed dataset descriptions, proper citations, and links to related academic papers. Additionally, your dataset gets its own SEO-optimized URL, making it easily discoverable by other researchers and practitioners.

### Tools for exploring and working with datasets

There are a growing number of tools being created which make it easier to understand datasets hosted on the Hugging Face Hub. 

### Tools for loading datasets hosted on the Hugging Face Hub 

Datasets shared on the Hugging Face Hub can be loaded via a variety of tools. The [`datasets`](https://huggingface.co/docs/datasets/) library is a Python library which can directly load datasets from the huggingface hub via a `load_dataset` command. The `datasets` library is optimized for working with large datasets (including datasets which won't fit into memory) and supporting machine learning workflows. 

Alongside this many of the datasets on the Hub can also be loaded directly into [`Pandas`](https://pandas.pydata.org/), [`Polars`](https://www.pola.rs/), and [`DuckDB`](https://duckdb.org/). This [page](https://huggingface.co/docs/datasets-server/parquet_process) provides a more detailed overview of the different ways you can load datasets from the Hub.


#### Datasets Viewer 

The datasets viewer allows people to explore and interact with datasets hosted on the Hub directly in the browser by visiting the dataset repository on the Hugging Face Hub. This makes it much easier for others to view and explore your data without first having to download it. The datasets viewer also allows you to search and filter datasets, which can be valuable to potential dataset users, understanding the nature of a dataset more quickly.


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/datasets-viewer.png" alt="Screenshot of a dataset viewer on the Hub showing a named entity recognition dataset"><br> 
<em>The dataset viewer for the multiconer_v2 Named Entity Recognition dataset.</em> 
 </p> 

### Community tools 

Alongside the datasets viewer there are a growing number of community created tools for exploring datasets on the Hub.

#### Spotlight

[`Spotlight`](https://github.com/Renumics/spotlight) is a tool that allows you to interactively explore datasets on the Hub with one line of code.   

<p align="center"><a href="https://github.com/Renumics/spotlight"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/scalable-data-inspection/speech_commands_vis_s.gif" width="100%"/></a></p> 

You can learn more about how you can use this tool in this [blog post](https://huggingface.co/blog/scalable-data-inspection).

#### Lilac

[`Lilac`](https://lilacml.com/) is a tool that aims to help you "curate better data for LLMs" and allows you to explore natural language datasets more easily. The tool allows you to semantically search your dataset (search by meaning), cluster data and gain high-level insights into your dataset.

<div style="text-align: center;">
    <iframe
        src="https://lilacai-lilac.hf.space"
        frameborder="0"
        width="850"
        height="450"
    ></iframe>
    <em>A Spaces demo of the lilac tool.</em> 
</div>

You can explore the `Lilac` tool further in a [demo](https://lilacai-lilac.hf.space/).

This growing number of tools for exploring datasets on the Hub makes it easier for people to explore and understand your datasets and can help promote your datasets to a wider audience.

### Support for large datasets

The Hub can host large datasets; it currently hosts datasets with multiple TBs of data.The datasets library, which users can use to download and process datasets from the Hub, supports streaming, making it possible to work with large datasets without downloading the entire dataset upfront. This can be invaluable for allowing researchers with less computational resources to work with your datasets, or to select small portions of a huge dataset for testing, development or prototyping.


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/filesize.png" alt="Screenshot of the file size information for a dataset"><br> 
<em>The Hugging Face Hub can host the large datasets often created for machine learning research.</em> 
 </p> 


## API and client library interaction with the Hub

Interacting with the Hugging Face Hub via an [API](https://huggingface.co/docs/hub/api) or the [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/index) Python library is possible. This includes creating new repositories, uploading data programmatically and creating and modifying metadata for datasets. This can be powerful for research workflows where new data or annotations continue to be created. The client library also makes uploading large datasets much more accessible. 

## Community 

The Hugging Face Hub is already home to a large community of researchers, developers, artists, and others interested in using and contributing to an ecosystem of open-source machine learning. Making your datasets accessible to this community increases their visibility, opens them up to new types of users and places your datasets within the context of a larger ecosystem of models, datasets and libraries.

The Hub also has features which allow communities to collaborate more easily. This includes a discussion page for each dataset, model and Space hosted on the Hub. This means users of your datasets can quickly ask questions and discuss ideas for working with a dataset. 


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/discussion.png" alt="Screenshot of a discussion for a dataset on the Hub."><br> 
<em>The Hub makes it easy to ask questions and discuss datasets.</em> 
 </p> 


### Other important features for researchers

Some other features of the Hub may be of particular interest to researchers wanting to share their machine learning datasets on the Hub:

- [Organizations](https://huggingface.co/organizations) allow you to collaborate with other people and share models, datasets and demos under a single organization. This can be an excellent way of highlighting the work of a particular research project or institute. 
- [Gated repositories](https://huggingface.co/docs/hub/datasets-gated) allow you to add some access restrictions to accessing your dataset. 
- Download metrics are available for datasets on the Hub; this can be useful for communicating the impact of your researchers to funders and hiring committees. 
- [Digital Object Identifiers (DOI)](https://huggingface.co/docs/hub/doi): it’s possible to register a persistent identifier for your dataset. 

### How can I share my dataset on the Hugging Face Hub? 

Here are some resources to help you get started with sharing your datasets on the Hugging Face Hub:

- General guidance on [creating](https://huggingface.co/docs/datasets/create_dataset) and [sharing datasets on the Hub](https://huggingface.co/docs/datasets/upload_dataset)
- Guides for particular modalities:
  - Creating an [audio dataset](https://huggingface.co/docs/datasets/audio_dataset)
  - Creating an [image dataset](https://huggingface.co/docs/datasets/image_dataset)
- Guidance on [structuring your repository](https://huggingface.co/docs/datasets/repository_structure) so a dataset can be automatically loaded from the Hub.

The following pages will be useful if you want to share large datasets:
- [Repository limitations and recommendations](https://huggingface.co/docs/hub/repositories-recommendations) provides general guidance on some of the considerations you'll want to make when sharing large datasets.
- The [Tips and tricks for large uploads](https://huggingface.co/docs/huggingface_hub/guides/upload#tips-and-tricks-for-large-uploads) page provides some guidance on how to upload large datasets to the Hub.

If you want any further help uploading a dataset to the Hub or want to upload a particularly large dataset, please contact datasets@huggingface.co. 
