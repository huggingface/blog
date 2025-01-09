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
# Visual Document Retrieval Goes Multilingual

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vdr-2b-multilingual/cover.png)

Introducing vdr-2b-multi-v1 [(🤗)](https://huggingface.co/llamaindex/vdr-2b-multi-v1), a multilingual embedding model designed for visual document retrieval across multiple languages and domains. This model is designed to encode document page screenshots into dense single-vector representations, this will effectively allow to search and query visually rich multilingual documents without the need for any OCR, data extraction pipelines, chunking...

vdr-2b-multi-v1 is based on [MrLight/dse-qwen2-2b-mrl-v1](https://huggingface.co/MrLight/dse-qwen2-2b-mrl-v1) and is trained on an extensive self-made dataset of multilingual query-image pairs. This model is built in collaboration with LlamaIndex and is the next iteration of [mcdse-2b-v1](https://huggingface.co/marco/mcdse-2b-v1). vdr-2b-multi-v1 extends and improves the learning and methods used to train it, resulting in a much more powerful and better model.

- **Trained on 🇮🇹 Italian, 🇪🇸 Spanish, 🇬🇧 English, 🇫🇷 French and 🇩🇪 German:** together they form a new large, open-source, multilingual training dataset of 500k high-quality samples.

- **Low VRAM and Faster Inference**: english model achieves better results on synthetic vidore benchmarks with just 30% of the base model image resolution. This results in 3x faster inference and much lower VRAM usage.

- **Cross-lingual Retrieval**: substantially better on real-world scenarios. For example, this allows for searching german documents with italian queries.

- **Matryoshka Representation Learning**: You can reduce the vectors size 3x and still keep 98% of the embeddings quality.

## Training dataset
Training good single-vector models for visual document retrieval requires high-quality data, but the current multimodal off-the-shelf datasets are very scarce and not multilingual.

So, I've spent a lot of time building it from scratch. The raw dataset consists of 500k multilingual query image samples, collected and generated from scratch using public internet pdfs. The queries associated with each image are synthetic and generated using VLMs. For comparison, it has 10x more samples than the largest open source synthetic dataset for multimodal visual document ir (colpali trainset scrapped documents).

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vdr-2b-multilingual/datapipeline.png)

#### Data gathering

For each language, I generate a long list of search queries covering many different topics, which are then used to search for PDFs. I use the language filtering capabilities of the search engine to scrape documents that are only in the specified language. This "search by topic" technique ensures that the model has seen a lot of diverse topics and domains, and that it performs well in real life scenarios.

The scraping process produced ~50k multilingual documents. Contrary to the method used in the previous mcdse-2b-v1 model, pages were not extracted randomly. Instead, each page of each PDF was run through a document layout analysis model to determine whether the page contained more textual or visual elements. The result is a number that classifies the page as text-only, visual-only or mixed. This labelling step was then used to sample ~100k pages, ensuring they were evenly distributed by page type.

#### Synthetic generation
The queries were then generated using gemini-1.5-pro and Qwen2-VL-72B. They were tasked to come up with a specific and a general question. Only the specific question is then used to train the model, but forcing it to distinguish between the two often made the specific questions better for information retrieval tasks.

After generation, a further cleaning step ensures that the questions are good enough for training. This includes:

- Ensuring the language is correct
- Fix formatting problems
- Remove markdown
- Ensuring that only one question is posed
- Removing grounding phrases (e.g. "according to Figure 1", "this document", ...)


#### Filtering and hard-negative mining

This cleaning step ensures that the queries are syntactically correct and follow some strict guidelines. But it still doesn't ensure that the queries are good enough for information retrieval.

To filter out bad questions, I have embedded and indexed each broad query with the voyage-3 model. For each specific question, I search the index. The query is marked as 'good' if its associated broad question appears in the top 100 results. This method removes low entropy, duplicate or too similar questions. On average, 40% of queries were removed from each language dataset.

Hard negatives were then mined using voyage-3 only on specific questions with a fixed threshold of 0.75. Experiments were also carried out using positive aware negative mining as described in [nvidia/NV-Retriever-v1](https://huggingface.co/nvidia/NV-Retriever-v1), but on this dataset it seems to produce too easy/distant negatives.

### Download

The training ([vdr-multilingual-train 🤗](https://huggingface.co/datasets/llamaindex/vdr-multilingual-train)) dataset is now open-source and directly available on HuggingFace. The training dataset is comprised of 496,167 unfiltered query->image pairs. Almost 40% of the queries were filtered out using the method described above.

|  Language | # filtered queries | # unfiltered queries |
|----------:|-------------------:|---------------------:|
|   English |             53,512 |               94,225 |
|   Spanish |             58,738 |              102,685 |
|   Italian |             54,942 |               98,747 |
|    German |             58,217 |              100,713 |
|    French |             55,270 |               99,797 |
| **TOTAL** |        **280,679** |          **496,167** |

The dataset is made of 5 different subsets, each for every language. You can download languages individually with:

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

The model has been evaluated on the Vidore benchmark and on custom-built evaluation sets that allow testing its multilingual capabilities on text-only, visual-only and mixed page screenshots. The evaluation dataset is also publicly available on HuggingFace ([vdr-multilingual-test 🤗](https://huggingface.co/datasets/llamaindex/vdr-multilingual-test)).

I made sure that no page in these datasets was also present in the training set to avoid any evaluation contamination. The datasets were collected and generated using the same methods as the training dataset, but with a smaller sample size. The filtering step was all done manually: each query is evaluated, curated and improved (if necessary) to ensure high data quality. 



All evaluations are performed by calculating **NDCG@5** scores using **1536 dimensions** vectors and an image resolution that can be represented with **maximum 768 tokens**. 

|                     | Avg      | Italian (text) | Italian (visual) | Italian (mix) |
|---------------------|----------|----------------|------------------|---------------|
| dse-qwen2-2b-mrl-v1 |     95.1 |           95.1 |               94 |          96.2 |
| vdr-2b-multi-v1     | **97.0** |       **96.4** |         **96.3** |      **98.4** |
|                     |  **+2%** |                |                  |               |

|                     | Avg       | French (text) | French (visual) | French (mix) |
|---------------------|-----------|---------------|-----------------|--------------|
| dse-qwen2-2b-mrl-v1 |      93.5 |          94.7 |            90.8 |         95.1 |
| vdr-2b-multi-v1     |  **95.6** |      **95.6** |        **93.3** |     **97.9** |
|                     | **+2.2%** |               |                 |              |

|                     | Avg       | Spanish (text) | Spanish (visual) | Spanish (mix) |
|---------------------|-----------|----------------|------------------|---------------|
| dse-qwen2-2b-mrl-v1 |      96.7 |           97.2 |             94.7 |          98.2 |
| vdr-2b-multi-v1     |  **98.1** |       **98.3** |         **96.9** |      **99.1** |
|                     | **+1.4%** |                |                  |               |

|                     | Avg       | German (text) | German (visual) | German (mix) |
|---------------------|-----------|---------------|-----------------|--------------|
| dse-qwen2-2b-mrl-v1 |      93.0 |          93.4 |              90 |         95.5 |
| vdr-2b-multi-v1     |  **96.2** |      **94.8** |        **95.7** |     **98.1** |
|                     | **+3.4%** |               |                 |              |

|                     | Avg       | English (text) | English (visual) | English (mix) |
|---------------------|-----------|----------------|------------------|---------------|
| dse-qwen2-2b-mrl-v1 | 98.0      | **98.3**       | 98.5             | 97.1          |
| vdr-2b-multi-v1     | **98.1**  | 97.9           | **99.1**         | **97.3**      |
|                     | **+0.1%** |                |                  |               |

The multilingual model outperforms the base model in every language and every page type, on average by +2.3%. On the vidore benchmark, it also performs slightly better (+0.5%). Although it still delivers very good NDCG@5 results, it's worth noting that the model performs worse on visual-only pages. However, this is also where the fine-tuned model improves the most, especially on German.

I also trained a version only on the English subset ([vdr-2b-v1 🤗](https://huggingface.co/llamaindex/vdr-2b-v1)). On the full Vidore benchmark (evaluated with 768 image tokens), both the multilingual and the english-only version performs better than the base model.

|                     | **Avg**  | **shiftproject** | **government** | **healthcare** | **energy** | **ai**   | **docvqa** | **arxivqa** | **tatdqa** | **infovqa** | **tabfquad** |
|---------------------|----------|------------------|----------------|----------------|------------|----------|------------|-------------|------------|-------------|--------------|
| dse-qwen2-2b-mrl-v1 | 83.6     | 79.8             | 95.7           | 96.9           | 92         | 98.2     | 56.3       | **85.2**    | 53.9       | 87.5        | 90.3         |
| vdr-2b-multi-v1     | 84.0     | 82.4             | 95.5           | 96.5           | 91.2       | **98.5** | **58.5**   | 84.7        | 53.6       | 87.1        | **92.2**     |
| vdr-2b-v1           | **84.3** | **83.4**         | **96.9**       | **97.2**       | **92.6**   | 96.8     | 57.4       | 85.1        | **54.1**   | **87.9**    | 91.3         |

![image/png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vdr-2b-multilingual/efficiency.png)

|                                         | Avg      | shiftproject | government | healthcare | energy   | ai       |
|-----------------------------------------|----------|--------------|------------|------------|----------|----------|
| dse-qwen2-2b-mrl-v1 (2560 image tokens) | 93.0     | 82           | 96         | 96.4       | **92.9** | **97.5** |
| vdr-2b-v1 (768 image tokens)            | **93.4** | **83.4**     | **96.9**   | **97.2**   | 92.6     | 96.8     |

vdr-2b-v1 also matches the performance of the base model on the Vidore benchmark synthetic datasets, while only using 30% of the image tokens (768 vs. 2560). This results in 3x faster inference and much lower VRAM usage.


### Cross-lingual retrieval

Although the model was trained on each language separately, it also improves in cross-lingual retrieval. To test this ability, the German evaluation set queries were translated into Italian using DeepL. The document page screenshots remain in the original German language.

|                     | Avg       | Italian -> German (text) | Italian -> German (visual) | Italian -> German (mix) |
|---------------------|-----------|--------------------------|----------------------------|-------------------------|
| dse-qwen2-2b-mrl-v1 | 93.1      | 92.6                     | 93.5                       | 93.3                    |
| vdr-2b-multi-v1     | **95.3**  | **95**                   | **95.8**                   | **95.1**                |
|                     | **+2.3%** |                          |                            |                         |


The model is significantly better across all document types, with an average improvement of +2.3%. These retrieval capabilities are essential for real-world use cases, especially in linguistically fragmented continents such as Europe. For example, it enables language-independent searches on complex multilingual sources such as European binding decisions, instruction manuals, financial asset KIDs, pharmaceutical package leaflets and many more...

### MRL and Binary Embeddings

This model is trained using Matryoshka Representation Learning (MRL). The loss function used during training is calibrated to track performance across all these dimensions, leading the model to frontload the most important identifying information. This effectively allows you to shrink the embedding dimensions according to your scale and budget.

To test the model retrieval capabilities with different vector dimensions, evaluations are performed in the Italian->German cross-lingual benchmark. 


#### NDCG@5 (float)
|                       | **Avg**   | **Italian -> German (text)** | **Italian -> German (visual)** | **Italian -> German (mix)** |
|-----------------------|-----------|------------------------------|--------------------------------|-----------------------------|
| **_1536 dimensions_** |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 93.1      | 92.6                         | 93.5                           | 93.3                        |
| vdr-2b-multi-v1       | **95.3**  | **95**                       | **95.9**                       | **95.1**                    |
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
| vdr-2b-multi-v1       | **90.8**  | **87**                       | **92.6**                       | **92.8**                    |
|                       | **+4.6%** |                              |                                |                             |
| **_512 dimensions_**  |           |                              |                                |                             |
| dse-qwen2-2b-mrl-v1   | 79.2      | **80.6**                     | 81.7                           | 75.4                        |
| vdr-2b-multi-v1       | **82.6**  | 77.7                         | **86.7**                       | **83.3**                    |
|                       | **+4.0%** |                              |                                |                             |

1024 dimension float vectors offer a very good balance between quality and size. They are ~30% smaller but still retain 99% of the retrieval performance. This is also true for the 1536 dimensions binary vectors, which have 10x fewer bytes per vector but still retain 97% of their retrieval quality. It's also interesting to see that 1536 binary vectors almost match the performance of the base model 1536 float vectors.

## Usage

Generating embeddings with vdr-2b-multi-v1 is easier than ever with SentenceTransformers and LlamaIndex direct integrations. Get started with just a few lines of code:

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
    model_name_or_path="llamaindex/vdr-2b-multi-v1",
    device="mps",
    trust_remote_code=True,
)

embeddings = model.get_image_embedding("image.png")
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
    device="mps",
    trust_remote_code=True,
    # These are the recommended kwargs for the model, but change them as needed
    model_kwargs={
        "torch_dtype": torch.bfloat16, 
        "device_map": "cuda:0", 
        "attn_implementation": "flash_attention_2"
    },
)

embeddings = model.encode("image.png")
```



</details>

## Next steps

While vdr-2b-multi-v1 performs well for English and multilingual retrieval overall, future work will explore how the model performs when adapted to new and specific domains. This is still in the early stages of development and more work needs to be done before I can publish the results, but early tests already seem to suggest impressive retrieval gains with **very** minimal data and computational resources. Stay tuned for future updates!