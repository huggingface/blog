---
title: "Visual Document Retrieval Goes Multilingual" 
thumbnail: /blog/assets/vdr-2b-multilingual/thumbnail.png
authors:
- user: marco
  guest: true
  org: llamaindex
- user: cheesyFishes
  guest: true
  org: llamaindex
---
> **TL;DR**: We present [`vdr-2b-multi-v1`](https://huggingface.co/llamaindex/vdr-2b-multi-v1), the best multilingual embedding model for visual document retrieval. We also release its English-only twin [`vdr-2b-v1`](https://huggingface.co/llamaindex/vdr-2b-v1) and open-source the new [`vdr-multilingual-train`](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train) dataset. With 500k high-quality samples, it's the largest open-source multilingual synthetic dataset for visual document retrieval.

# Visual Document Retrieval Goes Multilingual

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vdr-2b-multilingual/cover.png)

Introducing [`vdr-2b-multi-v1` (🤗)](https://huggingface.co/llamaindex/vdr-2b-multi-v1), a multilingual embedding model designed for visual document retrieval across multiple languages and domains. This model is designed to encode document page screenshots into dense single-vector representations, this will effectively allow to search and query visually rich multilingual documents without the need for any OCR, data extraction pipelines, chunking...

The `vdr-2b-multi-v1` model is based on [`MrLight/dse-qwen2-2b-mrl-v1`](https://huggingface.co/MrLight/dse-qwen2-2b-mrl-v1) and is trained on an extensive self-made dataset of multilingual query-image pairs. This model is built in collaboration with LlamaIndex and is the next iteration of [`mcdse-2b-v1`](https://huggingface.co/marco/mcdse-2b-v1). Our `vdr-2b-multi-v1` extends and improves the learning and methods used to train it, resulting in a much more powerful and better model.

- **Trained on 🇮🇹 Italian, 🇪🇸 Spanish, 🇬🇧 English, 🇫🇷 French and 🇩🇪 German:** Together they form a new large, open-source, multilingual training dataset of 500k high-quality samples.

- **Low VRAM and Faster Inference**: On synthetic Visual Document Retrieval (ViDoRe) benchmarks, our English-only model with 768 image patches performs better than the base model with 2560 image patches. This results in 3x faster inference and much lower VRAM usage.

- **Cross-lingual Retrieval**: Substantially better on real-world scenarios. For example, you can search for German documents with Italian queries.

- **Matryoshka Representation Learning**: You can reduce the vectors size 3x and still keep 98% of the embeddings quality. This allows for notably faster retrieval speeds while reducing storage costs.

## Usage

> 🎲 Try out `vdr-2b-multi-v1` now, available on this [Hugging Face Space](https://huggingface.co/spaces/llamaindex/multimodal_vdr_demo)!

Generating embeddings with `vdr-2b-multi-v1` is easier than ever with SentenceTransformers and LlamaIndex direct integrations. Get started with just a few lines of code:

<details open>
<summary>
via LlamaIndex
</summary>

```bash
pip install -U llama-index-embeddings-huggingface
```

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model = HuggingFaceEmbedding(
    model_name="llamaindex/vdr-2b-multi-v1",
    device="cpu",  # "mps" for mac, "cuda" for nvidia GPUs
    trust_remote_code=True,
)

image_embedding = model.get_image_embedding("image.png")
query_embedding = model.get_query_embedding("Chi ha inventato Bitcoin?")
```

</details>

<details>
<summary>
via SentenceTransformers
</summary>

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    model_name_or_path="llamaindex/vdr-2b-multi-v1",
    device="cuda",
    trust_remote_code=True,
    # These are the recommended kwargs for the model, but change them as needed if you don't have CUDA
    model_kwargs={
        "torch_dtype": torch.bfloat16, 
        "device_map": "cuda:0", 
        "attn_implementation": "flash_attention_2"
    },
)

embeddings = model.encode("image.png")
```



</details>

## Training Dataset
Training good single-vector models for visual document retrieval requires high-quality data, but the current multimodal off-the-shelf datasets are very scarce and not multilingual.

So, we've spent a lot of time building it from scratch. The raw dataset consists of 500k multilingual query-image samples, collected and generated from scratch using public internet PDFs. The queries associated with each image are synthetic and generated using VLMs. For comparison, our dataset has 10x more samples than the prior largest open source synthetic dataset for multimodal visual document retrieval, i.e. the scrapped documents generated for the [ColPali training dataset](https://huggingface.co/datasets/vidore/colpali_train_set).

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vdr-2b-multilingual/datapipeline.png)

### Data Gathering

For each language, we generate a long list of search queries covering many different topics, which are then used to search for PDFs. We use the language filtering capabilities of the search engine to scrape documents that are only in the specified language. This "search by topic" technique ensures that the model has seen a lot of diverse topics and domains, and that it performs well in real life scenarios.

The scraping process produced ~50k multilingual documents. Contrary to the method used in the previous [`mcdse-2b-v1`](https://huggingface.co/marco/mcdse-2b-v1) model, pages were not extracted randomly. Instead, each page of each PDF was run through a document layout analysis model to determine whether the page contained more textual or visual elements. The result is a number that classifies the page as text-only, visual-only or mixed. This labelling step was then used to sample ~100k pages, ensuring they were evenly distributed by page type.

### Synthetic Generation
The queries were then generated using gemini-1.5-pro and Qwen2-VL-72B. They were tasked to come up with a specific and a general question. Only the specific question is then used to train the model, but forcing the LLM to distinguish between the two often resulted in stronger specific questions for information retrieval training.

After generation, a further cleaning step ensures that the questions are good enough for training. This includes:

- Ensuring the language is correct
- Fix formatting problems
- Remove markdown
- Ensuring that only one question is posed
- Removing grounding phrases (e.g. "according to Figure 1", "this document", ...)


### Filtering and Hard-Negative Mining

This cleaning step ensures that the queries are syntactically correct and follow some strict guidelines. But it still doesn't ensure that the queries are good enough for information retrieval.

To filter out bad questions, we have embedded and indexed each broad query with the voyage-3 embedding model. For each specific question, we search the index. The query is marked as 'good' if its associated broad question appears in the top 100 results. This method removes low entropy, duplicate or too similar questions. On average, 40% of queries were removed from each language dataset.

Hard negatives were then mined using voyage-3 only on specific questions with a fixed threshold of 0.75. Experiments were also carried out using positive aware negative mining as described in [nvidia/NV-Retriever-v1](https://huggingface.co/nvidia/NV-Retriever-v1), but on this dataset it seems to produce too easy/distant negatives.

### Download

The ([vdr-multilingual-train 🤗](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train)) training dataset is now open-source and directly available on Hugging Face. The training dataset consists of 496,167 PDF pages, of which only 280,679 are associated with the filtered queries (using the method described above). The images that remain without a query are still used as hard negatives.

|  Language | # filtered queries | # unfiltered queries |
|----------:|-------------------:|---------------------:|
|   English |             53,512 |               94,225 |
|   Spanish |             58,738 |              102,685 |
|   Italian |             54,942 |               98,747 |
|    German |             58,217 |              100,713 |
|    French |             55,270 |               99,797 |
| **TOTAL** |        **280,679** |          **496,167** |

The dataset is made of 5 different subsets, each for every language. Here you can explore it directly:

<iframe
  src="https://huggingface.co/datasets/llamaindex/vdr-multilingual-train/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Alternatively, you can download languages individually by specifying the language subset in [`load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset):

```python
from datasets import load_dataset

italian_dataset = load_dataset("llamaindex/vdr-multilingual-train", "it", split="train")

english_dataset = load_dataset("llamaindex/vdr-multilingual-train", "en", split="train")

french_dataset = load_dataset("llamaindex/vdr-multilingual-train", "fr", split="train")

german_dataset = load_dataset("llamaindex/vdr-multilingual-train", "de", split="train")

spanish_dataset = load_dataset("llamaindex/vdr-multilingual-train", "es", split="train")
```

## Evaluations

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vdr-2b-multilingual/ndcgtop.png)

The model has been evaluated on the ViDoRe benchmark and on custom-built evaluation sets that allow testing its multilingual capabilities on text-only, visual-only and mixed page screenshots. The evaluation dataset is also publicly available on Hugging Face ([vdr-multilingual-test 🤗](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test)).

We made sure that no page in these datasets was also present in the training set to avoid any evaluation contamination. The datasets were collected and generated using the same methods as the training dataset, but with a smaller sample size. The filtering step was all done manually: each query is evaluated, curated and improved (if necessary) to ensure high data quality. 



All evaluations are performed by calculating **NDCG@5** scores using **1536 dimensions** vectors and an image resolution that can be represented with **maximum 768 tokens**. 

|                     | Avg       | French (text) | French (visual) | French (mix) |
|---------------------|-----------|---------------|-----------------|--------------|
| dse-qwen2-2b-mrl-v1 |      93.5 |          94.7 |            90.8 |         95.1 |
| vdr-2b-multi-v1     |  **95.6** |      **95.6** |        **93.3** |     **97.9** |
|                     | **+2.2%** |               |                 |              |

|                     | Avg       | German (text) | German (visual) | German (mix) |
|---------------------|-----------|---------------|-----------------|--------------|
| dse-qwen2-2b-mrl-v1 |      93.0 |          93.4 |            90.0 |         95.5 |
| vdr-2b-multi-v1     |  **96.2** |      **94.8** |        **95.7** |     **98.1** |
|                     | **+3.4%** |               |                 |              |

|                     | Avg      | Italian (text) | Italian (visual) | Italian (mix) |
|---------------------|----------|----------------|------------------|---------------|
| dse-qwen2-2b-mrl-v1 |     95.1 |           95.1 |             94.0 |          96.2 |
| vdr-2b-multi-v1     | **97.0** |       **96.4** |         **96.3** |      **98.4** |
|                     |  **+2%** |                |                  |               |

|                     | Avg       | Spanish (text) | Spanish (visual) | Spanish (mix) |
|---------------------|-----------|----------------|------------------|---------------|
| dse-qwen2-2b-mrl-v1 |      96.7 |           97.2 |             94.7 |          98.2 |
| vdr-2b-multi-v1     |  **98.1** |       **98.3** |         **96.9** |      **99.1** |
|                     | **+1.4%** |                |                  |               |

|                     | Avg       | English (text) | English (visual) | English (mix) |
|---------------------|-----------|----------------|------------------|---------------|
| dse-qwen2-2b-mrl-v1 | 98.0      | **98.3**       | 98.5             | 97.1          |
| vdr-2b-multi-v1     | **98.1**  | 97.9           | **99.1**         | **97.3**      |
|                     | **+0.1%** |                |                  |               |

The multilingual model outperforms the base model in every language and every page type, on average by +2.3%. On the ViDoRe benchmark, it also performs slightly better (+0.5%).
Our fine-tuned `vdr-2b-multi-v1` makes big leaps in performance, especially in non-English visual-only or mixed pages. See for example the +6.33% NDCG@5 improvement for German visual-only retrieval over the base model.

We also trained a version only on the English subset ([vdr-2b-v1 🤗](https://huggingface.co/llamaindex/vdr-2b-v1)). On the full ViDoRe benchmark (evaluated with 768 image tokens), both the multilingual and English-only versions outperform the base model.

|                     | **Avg**  | **shiftproject** | **government** | **healthcare** | **energy** | **ai**   | **docvqa** | **arxivqa** | **tatdqa** | **infovqa** | **tabfquad** |
|---------------------|----------|------------------|----------------|----------------|------------|----------|------------|-------------|------------|-------------|--------------|
| dse-qwen2-2b-mrl-v1 | 83.6     | 79.8             | 95.7           | 96.9           | 92.0       | 98.2     | 56.3       | **85.2**    | 53.9       | 87.5        | 90.3         |
| vdr-2b-multi-v1     | 84.0     | 82.4             | 95.5           | 96.5           | 91.2       | **98.5** | **58.5**   | 84.7        | 53.6       | 87.1        | **92.2**     |
| vdr-2b-v1           | **84.3** | **83.4**         | **96.9**       | **97.2**       | **92.6**   | 96.8     | 57.4       | 85.1        | **54.1**   | **87.9**    | 91.3         |

### Faster Inference
![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vdr-2b-multilingual/efficiency.png)

The English-only `vdr-2b-v1` model also matches the performance of the base model on the ViDoRe benchmark synthetic datasets, while only using 30% of the image tokens (768 vs. 2560). This effectively results in 3x faster inference and much lower VRAM usage.

|                                         | Avg      | shiftproject | government | healthcare | energy   | ai       |
|-----------------------------------------|----------|--------------|------------|------------|----------|----------|
| dse-qwen2-2b-mrl-v1 (2560 image tokens) | 93.0     | 82           | 96         | 96.4       | **92.9** | **97.5** |
| vdr-2b-v1 (768 image tokens)            | **93.4** | **83.4**     | **96.9**   | **97.2**   | 92.6     | 96.8     |




### Cross-Lingual Retrieval

Although the model was trained on each language separately, it also improves in cross-lingual retrieval. To test this ability, the German evaluation set queries were translated into Italian using DeepL. The document page screenshots remain in the original German language.

|                     | Avg       | Italian -> German (text) | Italian -> German (visual) | Italian -> German (mix) |
|---------------------|-----------|--------------------------|----------------------------|-------------------------|
| dse-qwen2-2b-mrl-v1 | 93.1      | 92.6                     | 93.5                       | 93.3                    |
| vdr-2b-multi-v1     | **95.3**  | **95.0**                 | **95.8**                   | **95.1**                |
|                     | **+2.3%** |                          |                            |                         |


The model is significantly better across all document types, with an average improvement of +2.3%. These retrieval capabilities are essential for real-world use cases, especially in linguistically fragmented continents such as Europe. For example, it enables language-independent searches on complex multilingual sources such as European binding decisions, instruction manuals, financial asset KIDs, pharmaceutical package leaflets and many more...

### MRL and Binary Embeddings

This model is trained using Matryoshka Representation Learning (MRL). The loss function used during training is calibrated to track performance across all these dimensions, leading the model to frontload the most important identifying information. This effectively allows you to shrink the embedding dimensions according to your scale and budget.
To learn more about MRL, [this blog post](https://huggingface.co/blog/matryoshka) by Hugging Face explains it very well.

To test the model retrieval capabilities with different vector dimensions, evaluations are performed in the Italian->German cross-lingual benchmark. 


#### NDCG@5 (float)
|                       | **Avg**   | **Italian -> German (text)** | **Italian -> German (visual)** | **Italian -> German (mix)** |
|-----------------------|-----------|------------------------------|--------------------------------|-----------------------------|
| **_1536 dimensions_** |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 93.1      | 92.6                         | 93.5                           | 93.3                        |
| vdr-2b-multi-v1       | **95.3**  | **95.0**                     | **95.9**                       | **95.1**                    |
|                       | **+2.3%** |                              |                                |                             |
| **_1024 dimensions_** |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 92.2      | 90.9                         | 92.3                           | 93.5                        |
| vdr-2b-multi-v1       | **94.6**  | **93.1**                     | **95.7**                       | **95.1**                    |
|                       | **+2.5%** |                              |                                |                             |
| **_512 dimensions_**  |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 89.8      | 87.9                         | 89.4                           | 92.2                        |
| vdr-2b-multi-v1       | **93.0**  | **91.1**                     | **93.4**                       | **94.5**                    |
|                       | **+3.4%** |                              |                                |                             |

#### NDCG@5 (binary)

|                       | **Avg**   | **Italian -> German (text)** | **Italian -> German (visual)** | **Italian -> German (mix)** |
|-----------------------|-----------|------------------------------|--------------------------------|-----------------------------|
| **_1536 dimensions_** |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 89.8      | 88.2                         | 90.3                           | 90.8                        |
| vdr-2b-multi-v1       | **92.3**  | **89.6**                     | **94.1**                       | **93.3**                    |
|                       | **+2.8%** |                              |                                |                             |
| **_1024 dimensions_** |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 86.7      | 84.9                         | 88.2                           | 86.9                        |
| vdr-2b-multi-v1       | **90.8**  | **87.0**                     | **92.6**                       | **92.8**                    |
|                       | **+4.6%** |                              |                                |                             |
| **_512 dimensions_**  |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 79.2      | **80.6**                     | 81.7                           | 75.4                        |
| vdr-2b-multi-v1       | **82.6**  | 77.7                         | **86.7**                       | **83.3**                    |
|                       | **+4.0%** |                              |                                |                             |

1024 dimension float vectors offer a very good balance between quality and size. They are ~30% smaller but still retain 99% of the retrieval performance. This is also true for the 1536 dimensions binary vectors, which have 10x fewer bytes per vector but still retain 97% of their retrieval quality. It's also interesting to see that 1536 binary vectors almost match the performance of the base model 1536 float vectors.

## Conclusions and Next Steps
We believe that [`vdr-2b-multi-v1`](https://huggingface.co/llamaindex/vdr-2b-multi-v1) and [`vdr-2b-v1`](https://huggingface.co/llamaindex/vdr-2b-v1) will prove useful to many users. 
 
Our multilingual model is the first of it's kind, it significantly improves performance in multilingual and cross-lingual scenarios, and thanks to MRL and Binary Quantization, retrieval is more efficient and faster than ever. We believe this will unlock new use-cases and opportunities, especially in linguistically fragmented continents such as Europe.

Its English-only twin represents a considerable improvement over the base model, now being able to embed documents 3x faster, with much less VRAM and with the same (or better) retrieval quality.

All this is possible thanks to the new [`vdr-multilingual-train`](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train) dataset. With 500k high quality samples, it is the largest multilingual open source synthetic dataset for visual document retrieval.

Future work will explore how our models perform when adapted to new and specific domains. This is still in the early stages of development and more work needs to be done before results are published, but early tests already seem to suggest impressive retrieval gains with **very** minimal data and computational resources.

Stay tuned for future updates!

## Links

* 🎲 Model demo: [Hugging Face Space](https://huggingface.co/spaces/llamaindex/multimodal_vdr_demo)
* 🤗 Multilingual model: [vdr-2b-multi-v1](https://huggingface.co/llamaindex/vdr-2b-multi-v1)
* 🤗 English-only model: [vdr-2b-v1](https://huggingface.co/llamaindex/vdr-2b-v1)
* 📂 Training dataset: [vdr-multilingual-train](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train)
* 📂 Evaluation dataset: [vdr-multilingual-test](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test)