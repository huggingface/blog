---
title: "Share your open ML datasets on Hugging Face Hub!"
thumbnail: /blog/assets/researcher-dataset-sharing/thumbnail.png
authors:
- user: davanstrien
- user: cfahlgren1
- user: lhoestq
- user: erinys
---

If you're working on data-intensive research or machine learning projects, you need a reliable way to share and host your datasets. Public datasets such as Common Crawl, MNIST, ImageNet, and more are critical to the open ML ecosystem, yet they can be challenging to host and share.

Hugging Face Hub makes it seamless to host and share datasets, trusted by many leading research institutions, companies, and government agencies, including [Meta](https://huggingface.co/meta-llama), [Google](https://huggingface.co/google), [Stanford](https://huggingface.co/stanfordnlp), NASA(https://huggingface.co/ibm-nasa-geospatial) and [THUDM](https://huggingface.co/THUDM).
 
By hosting a dataset on the Hugging Face Hub, you get instant access to features that can maximize your work's impact:

- [Generous Limits](#generous-limits)
- [Dataset Viewer](#dataset-viewer)
- [Third Party Library Support](#third-party-library-support)
- [SQL Console](#sql-console)
- [Security](#security)
- [Reach and Visibility](#reach-and-visibility)

##  Generous Limits

### Support for large datasets

The Hub can host terabyte-scale datasets, with high [per-file and per-repository limits](https://huggingface.co/docs/hub/en/repositories-recommendations). If you have data to share, the Hugging Face datasets team can help suggest the best format for uploading your data for community usage. 
The [🤗 Datasets library](https://huggingface.co/docs/datasets/index) makes it easy to upload and download your files, or even create a dataset from scratch. 🤗 Datasets also enables dataset streaming , making it possible to work with large datasets without needing to download the entire thing. This can be invaluable to allow researchers with less computational resources to work with your datasets, or to select small portions of a huge dataset for testing, development or prototyping.


<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/filesize.png" alt="Screenshot of the file size information for a dataset"><br> 
<em>The Hugging Face Hub can host the large datasets often created for machine learning research.</em> 
 </p> 

_Note: The [Xet team](https://huggingface.co/xet-team) is currently working on a backend update that will increase per-file limits from the current 50 GB to 500 GB while also improving storage and transfer efficiency._ 

## Dataset Viewer

Beyond just hosting your data, the Hub provides powerful tools for exploration. With the Datasets Viewer, users can explore and interact with datasets hosted on the Hub directly in their browser. This provides an easy way for others to view and explore your data without downloading it first.

Hugging Face datasets supports many different modalities (audio, images, video, etc.) and file formats (CSV, JSON, Parquet, etc.), and compression formats (Gzip, Zip, etc.). Check out the [Datasets File Formats](https://huggingface.co/docs/hub/en/datasets-adding#file-formats) page for more details.

<p align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/infinity-instruct.png" alt="Screenshot of the Datasets Viewer"><br> 
<em>The Dataset Viewer for the Infinity-Instruct dataset.</em> 
</p> 

The Datasets Viewer also includes a few features which make it easier to explore a dataset.

### Full Text Search

Built-in Full Text Search is one of the most powerful features of the Datasets Viewer. Any text columns in a dataset immediately become searchable.

The Arxiver dataset contains 63.4k rows of arXiv research papers converted to Markdown. By using Full Text Search, it's easy to find the papers containing a specific author such as Ilya Sutskever below.

<iframe
  src="https://huggingface.co/datasets/neuralwork/arxiver/embed/viewer/default/train?q=ilya+sutskever"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### Sorting

The Datasets Viewer allows you to sort the dataset by clicking on the column headers. This makes it easy to find the most relevant examples in a dataset.

Below is an example of a dataset sorted by the `helpfulness` column in descending order for the [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) dataset.

<iframe
  src="https://huggingface.co/datasets/nvidia/HelpSteer2/embed/viewer/default/train?sort[column]=helpfulness&sort[direction]=desc"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

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
| [WebDataset](https://huggingface.co/docs/hub/datasets-webdataset) | Library to write I/O pipelines for large datasets. | **871K** |
| [Argilla](https://huggingface.co/docs/hub/datasets-argilla) | Collaboration tool for AI engineers and domain experts that value high quality data. | **400k** |

 Most of these libraries enable you to load or stream a dataset in 1 single line of code. 
 
 Here's pandas as an example:

```python
# Pandas example
import pandas as pd
df = pd.read_parquet("hf://datasets/neuralwork/arxiver/data/train.parquet")

# Polars example 
import polars as pl
df = pl.read_parquet("hf://datasets/neuralwork/arxiver/data/train.parquet")

# DuckDB example - SQL queries
import duckdb
duckdb.sql("SELECT * FROM 'hf://datasets/neuralwork/arxiver/data/train.parquet' LIMIT 10")
```

You can find more information about integrated libraries in the [Datasets documentation](https://huggingface.co/docs/hub/en/datasets-libraries). Along with the libraries listed above, there are many more community supported tools which support the Hugging Face Hub such as [Lilac](https://lilacml.com/) and [Spotlight](https://github.com/Renumics/spotlight).

## SQL Console

The [SQL Console](https://huggingface.co/blog/sql-console) provides an interactive SQL editor that runs entirely in your browser, enabling instant data exploration without any setup. Key features include:

- **One-Click**: Open a SQL Console to query a dataset with a single click
- **Shareable and Embeddable Results**: Share and embed interesting query results
- **Full DuckDB Syntax**: Use full SQL syntax with built-in functions for regex, lists, JSON, embeddings, and more

On every public dataset you should see a new **SQL Console** badge. With just one click you can open a SQL Console to query that dataset.

<figure class="image flex flex-col items-center text-center m-0 w-full">
   <video
      alt="SQL Console Demo"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/Magpie-Ultra-Demo-SQL-Console.mp4" type="video/mp4">
  </video>
  <figcaption class="text-center text-sm italic">Querying the Magpie-Ultra dataset for excellent, high quality reasoning instructions.</figcaption>
</figure>


## Security

While making datasets accessible is important, protecting sensitive data is equally crucial. The Hugging Face Hub provides robust security features to help you maintain control over your data while sharing it with the right audiences.

### Access Controls

The Hugging Face Hub supports unique access control options for who has access to the dataset.

- **Public**: Anyone can access the dataset.
- **Private**: Only you and those in your organization can access the dataset.
- **Gated**: Control access to your dataset through two options:
  - **Automatic Approval**: Users must provide required information (like name and email) and agree to terms before gaining access
  - **Manual Approval**: You review and manually approve/reject each access request

For more details about gated datasets, see the [gated datasets documentation](https://huggingface.co/docs/hub/en/datasets-gated). For more fine-grained controls, there are Enterprise plan features where organizations can [create resource security groups](https://huggingface.co/docs/hub/en/security-resource-groups), use [SSO](https://huggingface.co/docs/hub/en/security-sso), and more.

### Built-in Security Scanning

Along with access controls, the Hugging Face Hub offers several security scanners:
| Feature | Description |
| :---- | :---- |
| [Malware Scanning](https://huggingface.co/docs/hub/en/security-malware) | Scans files for malware and suspicious content at each commit and visit |
| [Secrets Scanning](https://huggingface.co/docs/hub/en/security-secrets) | Blocks datasets with hardcoded secrets and environment variables |
| [Pickle Scanning](https://huggingface.co/docs/hub/en/security-pickle) | Scans pickle files and shows vetted imports for PyTorch weights |
| [ProtectAI](https://huggingface.co/docs/hub/en/security-protectai) | Uses Guardian tech to block datasets with pickle, Keras and other exploits |

<figure class="flex flex-col items-center text-center m-0 w-full">
  <img class="rounded" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/security-scanner-status-banner.png" alt="Security scanner status banner showing various security checks"/>
  <figcaption class="text-center text-sm italic mt-2">To learn more about these scanners, see the <a href="https://huggingface.co/docs/hub/en/security">security scanners documentation</a>.</figcaption>
</figure>


## Reach and Visibility

Having a secure platform with powerful features is valuable, but the true impact of research comes from reaching the right audience. Reach and visibility are crucial for researchers sharing datasets - it helps maximize research impact, enables reproducibility, facilitates collaboration, and ensures valuable data can benefit the broader scientific community.

With over 5M builders actively using the platform, the Hugging Face Hub provides researchers with powerful tools for community engagement and visibility. Here's what you can expect:

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

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/discussion.png" alt="Screenshot of a discussion for a dataset on the Hub."><br> 
<em>The Hub makes it easy to ask questions and discuss datasets.</em> 
 </p> 

## How can I host my dataset on the Hugging Face Hub? 

Now that you understand the benefits of hosting your dataset on the Hub, you might be wondering how to get started. Here are some comprehensive resources to guide you through the process:

- General guidance on [creating](https://huggingface.co/docs/datasets/create_dataset) and [sharing datasets on the Hub](https://huggingface.co/docs/datasets/upload_dataset)
- Guides for particular modalities:
  - Creating an [audio dataset](https://huggingface.co/docs/datasets/audio_dataset)
  - Creating an [image dataset](https://huggingface.co/docs/datasets/image_dataset)
  - Creating a [video dataset](https://huggingface.co/docs/datasets/main/en/video_dataset)
- Guidance on [structuring your repository](https://huggingface.co/docs/datasets/repository_structure) so a dataset can be automatically loaded from the Hub.

The following pages will be useful if you want to share large datasets:
- [Repository limitations and recommendations](https://huggingface.co/docs/hub/repositories-recommendations) provides general guidance on some of the considerations you'll want to make when sharing large datasets.
- The [Tips and tricks for large uploads](https://huggingface.co/docs/huggingface_hub/guides/upload#tips-and-tricks-for-large-uploads) page provides some guidance on how to upload large datasets to the Hub.

If you want any further help uploading a dataset to the Hub or want to upload a particularly large dataset, please contact datasets@huggingface.co. 
