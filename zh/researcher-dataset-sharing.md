---
title: "在 Hugging Face Hub 分享你的开源数据集"
thumbnail: /blog/assets/researcher-dataset-sharing/thumbnail.png
authors:
- user: davanstrien
- user: cfahlgren1
- user: lhoestq
- user: erinys
translators:
- user: AdinaY
---

### 在 Hugging Face Hub 上分享你的开源数据集！

如果您正在从事数据密集型研究或机器学习项目，那么您需要一种可靠的方式来共享和托管数据集。公共数据集（如 Common Crawl、ImageNet、Common Voice 等）对开放的机器学习生态系统至关重要，但它们往往难以托管和分享。

Hugging Face Hub 使得托管和共享数据集的流程变得更为流畅。许多顶尖研究机构、公司和政府机构（包括 [Nvidia](https://huggingface.co/nvidia)、[Google](https://huggingface.co/google)、[Stanford](https://huggingface.co/stanfordnlp)、[NASA](https://huggingface.co/ibm-nasa-geospatial)、[THUDM](https://huggingface.co/THUDM) 和 [Barcelona Supercomputing Center](https://huggingface.co/BSC-LT)）等都在使用。

在 Hugging Face Hub 上托管数据集，您将立即获得以下功能，从而最大化您的工作影响力：

- [慷慨的限制](#慷慨的限制)
- [数据集查看器](#数据集查看器)
- [第三方库支持](#第三方库支持)
- [SQL 控制台](#sql-控制台)
- [安全性](#安全性)
- [覆盖范围与可见性](#覆盖范围与可见性)

## 灵活的容量支持

### 支持大容量数据集

Hub 可托管 TB 级的数据集，并提供高 [单文件和单库限制](https://huggingface.co/docs/hub/en/repositories-recommendations)。如果您需要分享数据，Hugging Face 数据集团队可以为您提供建议，帮助您找到最佳格式以供社区使用。  
[🤗 Datasets 库](https://huggingface.co/docs/datasets/index) 使上传和下载文件，甚至从头创建数据集变得容易。🤗 Datasets 还支持数据流处理，使得无需下载整个数据集即可使用大规模数据。这对研究资源有限的研究人员来说尤为重要，还能支持从巨大数据集中选择小部分用于测试、开发或原型制作。

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/filesize.png" alt="数据集文件大小信息的截图"><br> 
<em>Hugging Face Hub 可托管机器学习研究中经常创建的大容量数据集。</em> 
 </p> 

_注意：[Xet 团队](https://huggingface.co/xet-team) 目前正在开发一项后台更新，将单文件限制从 50 GB 提高到 500 GB，同时提升存储和传输效率。_ 

## 数据集查看器

除了托管数据，Hub 还提供强大的检索工具。通过数据集查看器，用户可以直接在浏览器中探索和交互 Hub 上的托管数据集，无需提前下载。这为其他人查看和检索数据提供了一种简单的方法。

Hugging Face 数据集支持多种模态（音频、图像、视频等）和文件格式（CSV、JSON、Parquet 等），以及压缩格式（Gzip、Zip 等）。查看 [数据集文件格式](https://huggingface.co/docs/hub/en/datasets-adding#file-formats) 页面了解更多信息。

<p align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/infinity-instruct.png" alt="数据集查看器截图"><br> 
<em>Infinity-Instruct 数据集的查看器。</em> 
</p> 

数据集查看器还包含一些其他功能，方便用户更轻松地检索数据集。

### 全文搜索

内置的全文搜索是数据集查看器最强大的功能之一。数据集中任何文本列都可以直接进行搜索。

例如，Arxiver 数据集包含 63.4k 条将 arXiv 研究论文转换为 Markdown 的记录。通过全文搜索，可以轻松找到包含特定作者（例如 Ilya Sutskever）的论文。

<iframe
  src="https://huggingface.co/datasets/neuralwork/arxiver/embed/viewer/default/train?q=ilya+sutskever"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### 排序

数据集查看器允许通过单击列标题对数据集进行排序。更容易在数据集中找到最相关的示例。

以下是 [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) 数据集中根据 `helpfulness` 列降序排序的示例。

<iframe
  src="https://huggingface.co/datasets/nvidia/HelpSteer2/embed/viewer/default/train?sort[column]=helpfulness&sort[direction]=desc"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## 第三方库支持

Hugging Face Hub 与主要开源数据工具拥有广泛的第三方集成。在 Hub 上托管数据集后，该数据集可以立即与用户最熟悉的工具兼容。

以下是 Hugging Face 原生支持的一些库：

| 库 | 描述 | 2024年每月 PyPi 下载量 |
| :---- | :---- | :---- |
| [Pandas](https://huggingface.co/docs/hub/datasets-pandas) | Python 数据分析工具包。 | **2.58 亿** |
| [Spark](https://huggingface.co/docs/hub/datasets-spark) | 分布式环境中的实时大规模数据处理工具。 | **2,900 万** |
| [Datasets](https://huggingface.co/docs/hub/datasets-usage) | 🤗 Datasets 是一个音频、计算机视觉和自然语言处理 (NLP) 的数据集库。 | **1,700 万** |
| [Dask](https://huggingface.co/docs/hub/datasets-dask) | 一个可扩展现有 Python 和 PyData 生态的并行与分布式计算库。 | **1,200 万** |
| [Polars](https://huggingface.co/docs/hub/datasets-polars) | 基于 OLAP 查询引擎的数据框库。 | **850 万** |
| [DuckDB](https://huggingface.co/docs/hub/datasets-duckdb) | 内存中的 SQL OLAP 数据库管理系统。 | **600 万** |
| [WebDataset](https://huggingface.co/docs/hub/datasets-webdataset) | 用于大规模数据集的 I/O 管道编写的库。 | **87.1 万** |
| [Argilla](https://huggingface.co/docs/hub/datasets-argilla) | 针对 AI 工程师和领域专家的协作工具，注重高质量数据。 | **40 万** |

多数这些库支持用一行代码加载或流处理数据集。以下是 Pandas、Polars 和 DuckDB 的示例：

```python
# Pandas 示例
import pandas as pd
df = pd.read_parquet("hf://datasets/neuralwork/arxiver/data/train.parquet")

# Polars 示例 
import polars as pl
df = pl.read_parquet("hf://datasets/neuralwork/arxiver/data/train.parquet")

# DuckDB 示例 - SQL 查询
import duckdb
duckdb.sql("SELECT * FROM 'hf://datasets/neuralwork/arxiver/data/train.parquet' LIMIT 10")
```

您可以在 [数据集文档](https://huggingface.co/docs/hub/en/datasets-libraries) 中找到更多关于集成库的信息。

## SQL 控制台

[SQL 控制台](https://huggingface.co/blog/sql-console) 提供了一个完全在浏览器中运行的交互式 SQL 编辑器，能够即时探索数据，无需任何设置。主要功能包括：

- **一键操作**：通过单击即可打开 SQL 控制台查询数据集
- **可共享和嵌入的结果**：分享和嵌入有趣的查询结果
- **完整的 DuckDB 语法**：支持正则表达式、列表、JSON、嵌入等内置函数的完整 SQL 语法

<figure class="image flex flex-col items-center text-center m-0 w-full">
   <video
      alt="SQL 控制台演示"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/Magpie-Ultra-Demo-SQL-Console.mp4" type="video/mp4">
  </video>
  <figcaption class="text-center text-sm italic">查询 Magpie-Ultra 数据集以获得高质量推理指令。</figcaption>
</figure>

## 安全性

在确保数据可访问性的同时，保护敏感数据同样重要。Hugging Face Hub 提供了强大的安全功能，帮助您在共享数据的同时保持对数据的控制权。

### 访问控制

Hugging Face Hub 支持独特的访问控制选项：

- **公开**：任何人都可以访问数据集。
- **私有**：仅您和组织内的成员可访问数据集。
- **限流**：通过两种选项控制对数据集的访问：
  - **自动批准**：用户需提供必要信息（如姓名和邮箱）并同意条款后才可访问
  - **手动批准**：您审核并手动批准/拒绝每个访问请求

关于限流数据集的更多详情，请参阅 [限流数据集文档](https://huggingface.co/docs/hub/en/datasets-gated)。

### 内置安全扫描

Hugging Face Hub 提供多种安全扫描器：
| 功能 | 描述 |
| :---- | :---- |
| [恶意软件扫描](https://huggingface.co/docs/hub/en/security-malware) | 每次提交和访问时扫描文件中的恶意软件和可疑内容 |
| [密钥扫描](https://huggingface.co/docs/hub/en/security-secrets) | 阻止包含硬编码密钥和环境变量的数据集 |
| [Pickle 扫描](https://huggingface.co/docs/hub/en/security-pickle) | 扫描 Pickle 文件并显示经过验证的 PyTorch 权重导入 |
| [ProtectAI](https://huggingface.co/docs/hub/en/security-protectai) | 使用 Guardian 技术阻止包含 Pickle、Keras 等漏洞的数据集 |

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/security-scanner-status-banner.png" alt="安全扫描状态横幅"><br> 
<em>要了解更多安全扫描器，请参阅 <a href="https://huggingface.co/docs/hub/en/security">安全扫描器文档</a>。</em> 
</p> 

## 覆盖范围与可见性

一个安全的平台和强大的功能是重要的，但研究的真正影响力来源于触达正确的目标受众。覆盖范围和可见性对于分享数据集的研究人员至关重要，这有助于最大化研究影响力、实现可重复性、促进协作，并确保有价值的数据可以惠及更广泛的科学社区。

在 Hugging Face Hub 上，您可以通过以下方式扩大您的影响力：

### 更好的社区参与
- 每个数据集内置讨论标签以促进社区互动
- 支持集中化组织多个数据集并开展协作
- 提供数据集使用和影响力的指标

### 更广的覆盖
- 可触达一个活跃的研究人员、开发者和从业者社区
- SEO 优化 URL，让您的数据集更易被发现
- 与模型、数据集和库生态系统集成，提升关联性
- 清晰展示您的数据集与相关模型、论文和演示之间的链接

### 改进的文档
- 支持自定义 README 文件以实现全面的文档说明
- 支持详细的数据集描述和学术引用
- 链接到相关的研究论文和出版物

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/discussion.png" alt="数据集的讨论截图"><br> 
<em>Hub 使得提问和讨论数据集变得轻松。</em> 
 </p> 

## 如何在 Hugging Face Hub 上托管我的数据集？

了解了在 Hub 上托管数据集的好处后，您可能会想知道如何开始。以下是一些全面的资源，指导您完成整个过程：

- 关于[创建](https://huggingface.co/docs/datasets/create_dataset)和[共享数据集](https://huggingface.co/docs/datasets/upload_dataset)的常规指南
- 针对特定模态的指南：
  - 创建 [音频数据集](https://huggingface.co/docs/datasets/audio_dataset)
  - 创建 [图像数据集](https://huggingface.co/docs/datasets/image_dataset)
  - 创建 [视频数据集](https://huggingface.co/docs/datasets/video_dataset)
- 关于[组织您的数据集库](https://huggingface.co/docs/datasets/repository_structure)以便可以自动从 Hub 加载的指南。

如果您想共享大数据集，以下页面将非常有用：
- [数据集库限制与推荐](https://huggingface.co/docs/hub/repositories-recommendations) 提供了共享大数据集时需要注意的一些常规指导。
- [上传大数据集的技巧和方法](https://huggingface.co/docs/huggingface_hub/guides/upload#tips-and-tricks-for-large-uploads) 页面提供了上传大数据集到 Hub 的实用建议。

如果您需要任何进一步帮助，或计划上传特别大的数据集，请联系 datasets@huggingface.co。
