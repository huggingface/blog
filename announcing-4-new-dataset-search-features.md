---
title: "Announcing 4 New Dataset Search Features" 
thumbnail: /blog/assets/hf-reinvents-dataset-search/thumbnail.jpg
authors:
- user: lhoestq
- user: severo
---

# Announcing 4 New Dataset Search Features

The AI and ML community is sharing more than 180,000 public datasets on The [Hugging Face Dataset Hub](https://huggingface.co/datasets) to train and evaluate models.
Researchers and engineers are using these datasets for various tasks, from training LLMs to chat with users to evaluating automatic speech recognition or computer vision systems.
Datasets discoverability and vizualisation are key challenges to let AI builders find, explore and transform datasets to fit their use cases.

At Hugging Face we are building the Dataset Hub to be the best place for the community to collaborate on AI datasets.
So we built tools like Dataset Search, Dataset Viewer, as well as a rich open source ecosystem of tools.
Today we are announcing 4 new features that will take Dataset Search on the Hub to the next level.

## Search by Modality

The modality of a dataset corresponds to the type of the data inside the dataset. For example the most common types of data on Hugging Face are text, image, audio and tabular data.

We released a set of filters that allows you to filter datasets that have one or several modalities among this list:

- Text
- Image
- Audio
- Tabular
- Time-Series
- 3D
- Video
- Geospatial

For example it is possible to look for datasets that contain both text and image data:

![search by modality example](TODO: add image link)

The modalities of each dataset are automatically detected based on files contents and extensions.

## Search by Size

We recently released a new feature in the Dataset Hub interface to show the number of rows of each dataset:

![number of rows of each dataset](TODO: add image link)

Following this, it is now possible to search dataset by number of rows by specifing a minimum and maximum number of rows:

![number of rows of each dataset](TODO: add image link)

This will let you look for datasets of small size to the biggest datasets that exist (for example the ones used to pretrain LLMs).

The number of rows information is available for all the datasets in [supported formats](https://huggingface.co/docs/hub/datasets-adding#file-formats).
Even for the biggest datasets that don't have the number of rows in their metadata, the total number of rows is estimated accurately based on the content of the first 5GB.

For example if you are looking at the datasets with the highest number of rows on Hugging Face, you can look for datasets with more than 1T (10<sup>12</sup>) rows:

![biggest datasets](TODO: add image link)

## Search by Format

The same dataset can be stored in many different formats.
For example text datasets are often in Parquet or JSON Lines but they could be in text files, and image datasets are often a single directory of images but they could be in [WebDataset format](https://huggingface.co/docs/hub/datasets-webdataset) (a format based on TAR archives).

Each format has its pros and cons.
For example Parquet offers nested data support unlike CSV, efficient filtering/analytics and good compression ratio, but accessing one specific row requires to decode a full row group.
Another example is WebDataset, which offers the highest data streaming speed but lacks some metadata such as the number of rows per file, which is often needed to efficiently distribute data in multi-node training setups.

The dataset format therefore indicates which use cases are favoured, and whether you will need to reformat the data to fit your needs.

Here you can see the datasets in WebDataset format:

![webdatasets](TODO: add image link)

## Search by Library

There are many good libraries and tools to load datasets and prepare them for training like Pandas, Dask, or the 🤗 Datasets library.
The Hugging Face Hub allows you to use your favorite tools and to filter datasets compatible with any library:

![pandas compatible datasets](TODO: add image link)

The dataset compatibility is based on the dataset format and size (e.g. Dask can load big JSON Lines dataset unlike Pandas which requires to load the full dataset in memory).
In addition to this, we also provide the code snippet to load any dataset in your favorite tool:

![load fineweb-edu in dask](TODO: add image link)

## Combine filters

Those 4 new Dataset Search tools can be used together and with the other existing filters like Language, Tasks, Licenses.
Combining those filters with the text search bar you can look for the specific dataset you are looking for:

![search for a webdataset of images of pdf](TODO: add image link)
