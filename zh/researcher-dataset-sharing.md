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

如果您正在从事数据密集型研究或机器学习项目，您需要一种可靠的方法来分享和托管数据集。像 Common Crawl、ImageNet 和 Common Voice 这样的公共数据集是开源机器学习生态系统的关键，但它们的托管和共享却充满挑战。

Hugging Face Hub 极大简化了数据集的托管与共享过程，广受顶尖研究机构、企业和政府机构的信赖。包括 [Nvidia](https://huggingface.co/nvidia)、[Google](https://huggingface.co/google)、[Stanford](https://huggingface.co/stanfordnlp)、[NASA](https://huggingface.co/ibm-nasa-geospatial)、[THUDM](https://huggingface.co/THUDM) 和 [巴塞罗那超级计算中心](https://huggingface.co/BSC-LT)等团队都在使用。

将数据集托管在 Hugging Face Hub 上，您可以立即解锁以下多种强大功能，全面提升你的工作影响力：

- [丰富的容量限制](#丰富的容量限制)
- [数据集查看器](#数据集查看器)
- [第三方库支持](#第三方库支持)
- [SQL 控制台](#sql-控制台)
- [安全性](#安全性)
- [覆盖范围和可见性](#覆盖范围和可见性)

---

### 丰富的容量限制

#### 支持大型数据集

Hugging Face Hub 可以托管 TB 级别的数据集，提供高 [单文件和单仓库的限制](https://huggingface.co/docs/hub/en/repositories-recommendations)。如果您有数据需要分享，Hugging Face 的数据集团队可以帮助建议最佳的上传格式以便于社区使用。 

[🤗 Datasets 库](https://huggingface.co/docs/datasets/index) 让上传和下载文件变得更简单，甚至可以从头开始创建数据集。🤗 Datasets 还支持数据集流式处理，让用户无需下载整个数据集即可使用。这对于资源有限的研究人员来说非常有用，可以帮助他们在测试、开发或原型设计时只选择大型数据集中的一小部分数据。

<p align="center"> 
 <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/researcher-dataset-sharing/filesize.png" alt="显示数据集文件大小的截图"><br> 
<em>Hugging Face Hub 能托管机器学习研究中常见的大型数据集。</em> 
 </p> 

_注意：[Xet 团队](https://huggingface.co/xet-team) 正在开发一项后端更新，将单文件限制从当前的 50 GB 提高到 500 GB，同时提升存储和传输效率。_

---

### 数据集查看器

除了托管数据，Hugging Face Hub 还提供了强大的数据探索工具。通过数据集查看器，用户可以直接在浏览器中探索和交互数据集，而无需下载。

Hugging Face 数据集支持多种模态（如音频、图像、视频等）和文件格式（如 CSV、JSON、Parquet 等），以及压缩格式（如 Gzip、Zip 等）。有关更多细节，请参阅 [数据集文件格式](https://huggingface.co/docs/hub/en/datasets-adding#file-formats) 页面。

<p align="center"> 
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/infinity-instruct.png" alt="数据集查看器的截图"><br> 
<em>Infinity-Instruct 数据集的查看器界面。</em> 
</p> 

#### 全文搜索

数据集查看器内置了全文搜索功能，支持快速检索数据集中任意文本列的内容。例如，在 Arxiver 数据集中包含了 63.4k 条 Markdown 格式的 arXiv 研究论文记录。通过搜索功能，可以轻松找到特定作者（如 Ilya Sutskever）相关的论文。

#### 排序功能

用户还可以通过点击列标题对数据集进行排序。例如，您可以按“helpfulness”列对 [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) 数据集降序排列，以快速找到最相关的样本。

---

### 第三方库支持

在 Hugging Face Hub 托管数据集后，它将自动与许多主流开源数据工具兼容。

以下是 Hugging Face 支持的一些常用库：

| 库 | 描述 | 2024 年每月 PyPi 下载量 |
| :---- | :---- | :---- |
| [Pandas](https://huggingface.co/docs/hub/datasets-pandas) | Python 数据分析工具包。 | **2.58 亿** |
| [Spark](https://huggingface.co/docs/hub/datasets-spark) | 分布式环境中的实时大规模数据处理工具。 | **2900 万** |
| [Datasets](https://huggingface.co/docs/hub/datasets-usage) | 🤗 Datasets 提供音频、计算机视觉和自然语言处理的数据集访问与共享功能。 | **1700 万** |
| [Dask](https://huggingface.co/docs/hub/datasets-dask) | 支持并行和分布式计算的库。 | **1200 万** |
| [Polars](https://huggingface.co/docs/hub/datasets-polars) | 构建在 OLAP 查询引擎上的 DataFrame 库。 | **850 万** |
| [DuckDB](https://huggingface.co/docs/hub/datasets-duckdb) | 一种嵌入式 SQL OLAP 数据库管理系统。 | **600 万** |
| [WebDataset](https://huggingface.co/docs/hub/datasets-webdataset) | 设计用于大型数据集 I/O 管道的库。 | **87.1 万** |
| [Argilla](https://huggingface.co/docs/hub/datasets-argilla) | 面向高质量数据的 AI 工程师与领域专家协作工具。 | **40 万** |

多数这些库只需一行代码即可加载或流式处理数据集。例如：

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

更多集成库信息，请访问 [Datasets 文档](https://huggingface.co/docs/hub/en/datasets-libraries)。

---

### SQL 控制台

[SQL 控制台](https://huggingface.co/blog/sql-console) 是一个交互式 SQL 编辑器，完全运行在浏览器中，无需任何设置即可即时探索数据。其关键特性包括：

- 一键打开 SQL 控制台
- 可分享和嵌入查询结果
- 支持完整 DuckDB SQL 语法，包括正则表达式、列表、JSON 和嵌入等功能

---

### 安全性

Hugging Face Hub 提供强大的安全功能，保护敏感数据的同时确保其共享的可控性。

#### 访问控制

您可以通过以下方式控制数据集访问权限：
- **公开**：任何人都可以访问。
- **私有**：仅您和您的组织成员可以访问。
- **访问限制**：通过手动或自动审批控制访问权限。

#### 内置安全扫描

Hugging Face Hub 还提供了多种安全扫描工具，包括恶意软件扫描、密钥扫描和 Pickle 扫描等。

---

### 覆盖范围和可见性

Hugging Face Hub 上有超过 500 万开发者使用平台，研究者可以通过以下方式提升数据集影响力：

- 内置讨论功能，便于与社区互动
- 数据集使用与影响力统计
- 易于发现的 SEO 优化 URL

---

### 如何在 Hugging Face Hub 上托管数据集？

您可以通过以下资源了解如何开始：
- [创建和共享数据集](https://huggingface.co/docs/datasets/create_dataset)
- [上传大型数据集的技巧](https://huggingface.co/docs/huggingface_hub/guides/upload#tips-and-tricks-for-large-uploads)

如需更多帮助，请联系 datasets@huggingface.co。
