---
title: "Creating open machine learning datasets? Share them on the Hugging Face Hub!"
thumbnail: /blog/assets/researcher-dataset-sharing/thumbnail.png
authors:
- user: davanstrien
- user: cfahlgren1
- user: lhoestq
- user: erinys
---

## Who is this blog post for?

Are you a researcher doing data-intensive research or using machine learning as a research tool? As part of this research, you have likely created datasets for training and evaluating machine learning models, and like many researchers, you may be sharing these datasets via Google Drive, OneDrive, or your own personal server. In this post, we’ll outline why Hugging Face Hub is the best place on the internet to host open datasets.

Many of the world's leading research institutions and companies such as Meta, Google, Stanford and THUDM are using the Hugging Face Hub to host their datasets. 

By hosting a dataset on the Hugging Face Hub, you instantly get access to many features that can maximize the impact of your dataset:

- [Generous Limits](#generous-limits)
- [Dataset Viewer](#dataset-viewer)
- [Third Party Library Support](#third-party-library-support)
- [SQL Console](#sql-console)
- [Access Controls](#access-controls)
- [Reach and Visibility](#reach-and-visibility)

##  Generous Limits

### Support for large datasets

The Hub can host large datasets; it currently hosts datasets with multiple TBs of data.The datasets library, which users can use to download and process datasets from the Hub, supports streaming, making it possible to work with large datasets without downloading the entire dataset upfront. This can be invaluable for allowing researchers with less computational resources to work with your datasets, or to select small portions of a huge dataset for testing, development or prototyping.


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/filesize.png" alt="Screenshot of the file size information for a dataset"><br> 
<em>The Hugging Face Hub can host the large datasets often created for machine learning research.</em> 
 </p> 


## Dataset Viewer

The Datasets Viewer allows people to explore and interact with datasets hosted on the Hub directly in the browser when visiting the dataset. This makes it much easier for others to view and explore your data without first having to download it. The Datasets Viewer also includes a few features which make it easier to explore a dataset.

### Full Text Search

### Sorting

## Third Party Library Support

Hugging Face is fortunate to have third party integrations with the leading open source data tools. By hosting a dataset on the Hub, it instantly makes the dataset compatible with the tools users are most familiar with.

Here are some of the libraries Hugging Face supports out of the box:

| Library | Description | Monthly PyPi Downloads |
| :---- | :---- | :---- |
| [Pandas](https://huggingface.co/docs/hub/datasets-pandas) | Python data analysis toolkit. | **258M** |
| [Datasets](https://huggingface.co/docs/hub/datasets-usage) | 🤗 Datasets is a library for accessing and sharing datasets for Audio, Computer Vision, and Natural Language Processing (NLP). | **17M** |
| [Dask](https://huggingface.co/docs/hub/datasets-dask) | Parallel and distributed computing library that scales the existing Python and PyData ecosystem. | **12M** |
| [Polars](https://huggingface.co/docs/hub/datasets-polars) | A DataFrame library on top of an OLAP query engine. | **8.5M** |
| [DuckDB](https://huggingface.co/docs/hub/datasets-duckdb) | In-process SQL OLAP database management system. | **6M** |
| [WebDataset](https://huggingface.co/docs/hub/datasets-webdataset) | Library to write I/O pipelines for large datasets. | **871k** |
| [Argilla](https://huggingface.co/docs/hub/datasets-argilla) | Collaboration tool for AI engineers and domain experts that value high quality data. | **400k** |

 Most of these libraries enable you to load or stream a dataset in 1 single line of code. 
 
 Here's pandas as an example:

```py
import pandas as pd

df = pd.read_parquet("hf://datasets/neuralwork/arxiver/data/train.parquet")
```

You can find more information about integrated libraries in the [Datasets documentation](https://huggingface.co/docs/hub/en/datasets-libraries). Along with the libraries listed above, there are many more community supported tools which support the Hugging Face Hub such as [Lilac](https://lilacml.com/) and [Spotlight](https://github.com/Renumics/spotlight).

## SQL Console

## Access Controls

## Reach and Visibility

The Hugging Face Hub has become the central platform for open machine learning collaboration, offering researchers a powerful way to share and promote their datasets. When you host your dataset on the Hub, you gain:

### Better Community Engagement
- Built-in discussion tabs for each dataset for community engagement
- Organizations as a centralized place for grouping and collaborating on multiple datasets
- Metrics for dataset usage and impact

### Wider Reach
- Access to a large, active community of researchers, developers, and practitioners
- SEO-optimized URLs making your dataset easily discoverable 
- Integration with the broader ecosystem of models, datasets, and libraries
- Clear links between your dataset and related models, papers, and demos

### Improved Documentation
- Customizable README files for comprehensive documentation
- Support for detailed dataset descriptions and proper academic citations
- Links to related research papers and publications

<!-- TODO: Replace with a better image showing community engagement  -->
<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/discussion.png" alt="Screenshot of a discussion for a dataset on the Hub."><br> 
<em>The Hub makes it easy to ask questions and discuss datasets.</em> 
 </p> 

### How can I host my dataset on the Hugging Face Hub? 

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
