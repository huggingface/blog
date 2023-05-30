---
title: "Introducing BERTopic Integration with Hugging Face Hub"
thumbnail: /blog/assets/145_bertopic/logo.png
authors:
- user: MaartenGr
  guest: true
- user: davanstrien
  guest: false
---

<h1> Introducing BERTopic Integration with Hugging Face Hub"</h1> 

<!-- {blog_metadata} -->
<!-- {authors} -->

We are thrilled to announce a significant update to the [BERTopic](https://maartengr.github.io/BERTopic) Python library, expanding its capabilities and further streamlining the workflow for topic modelling enthusiasts and practitioners. BERTopic now supports pushing and pulling trained topic models directly to and from the Hugging Face Hub. This new integration opens up exciting possibilities for leveraging the power of BERTopic in production use cases with ease.

## What is BERTopic?

BERTopic is a state-of-the-art Python library that simplifies the topic modelling process using various embedding techniques and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.


<figure class="image table text-center m-0 w-full">
    <video 
        alt="BERTopic overview"
        style="max-width: 70%; margin: auto;"
        autoplay loop autobuffer muted playsinline
    >
      <source src="assets/145_bertopic/bertopic_overview.mp4" type="video/mp4">
  </video>
</figure>

*An overview of the BERTopic library*

BERTopic supports guided, supervised, semi-supervised, manual, long-document, hierarchical, class-based, dynamic, online, multimodal, and multi-aspect topic modeling. It even supports visualizations similar to LDAvis!

BERTopic provides a powerful tool for users to uncover significant topics within text collections, thereby gaining valuable insights. With BERTopic, users can analyze customer reviews, explore research papers, or categorize news articles with ease, making it an essential tool for anyone looking to extract meaningful information from their text data.

## BERTopic Model Management with the Hugging Face Hub

With the latest integration, BERTopic users can seamlessly push and pull their trained topic models to and from the Hugging Face Hub. This integration marks a significant milestone in simplifying the deployment and management of BERTopic models across different environments.

By leveraging the power of the Hugging Face Hub, BERTopic users can effortlessly share, version, and collaborate on their topic models. The Hub acts as a central repository, allowing users to store and organize their models, making it easier to deploy models in production, share them with colleagues, or even showcase them to the broader NLP community. Due to the improved saving procedure, training on large datasets generates small model sizes. In the example below, a BERTopic model was trained on 100.000 documents, resulting in a ~50MB model keeping all of the original’s model functionality. For inference, the model can be further reduced to only ~3MB!

TODO update image

https://raw.githubusercontent.com/MaartenGr/BERTopic/d58b5a41cfbc032cf6919697b133c840825bcc37/docs/getting_started/serialization/serialization.png #TODO update 

The benefits of this integration are particularly notable for production use cases. Users can now effortlessly deploy BERTopic models into their existing applications or systems, ensuring seamless integration within their data pipelines. This streamlined workflow enables faster iteration and efficient model updates and ensures consistency across different environments.

## safetensors: Ensuring Secure Model Management

In addition to the Hugging Face Hub integration, BERTopic now supports serialization using the safetensors library. Safetensors is a new simple format for storing tensors safely (instead of pickle), which is still fast (zero-copy). We’re excited to see more and more libraries leveraging safetensors for storing embeddings. You can read more about a recent audit of the library in this blog post. 

## Example: Using BERTopic to explore RLFH datasets

The last year has seen several datasets for Reinforcement Learning with Human Feedback released. One of these datasets is the [OpenAssistant Conversations dataset](https://huggingface.co/datasets/OpenAssistant/oasst1). This dataset was produced via a worldwide crowd-sourcing effort involving over 13,500 volunteers. Whilst this dataset already has some scores for toxicity, quality, humour etc., we want to get a better understanding of what types of conversations are represented in this dataset. 

BERTopic offers one way of getting a better understanding of the topics in this dataset. In this case, we train a model on the English assistant responses part of the datasets. Resulting in a [topic model](https://huggingface.co/davanstrien/chat_topics) with 75 topics. 

BERTopic gives us various ways of visualizing a dataset. We can see the top 8 topics and their associated words below. We can see that the second most frequent topic consists mainly of ‘response words’, which we often see frequently from chat models, i.e. responses which aim to be ‘polite’ and ‘helpful’. We can also see a large number of topics related to programming or computing topics as well as physics, recipes and pets. 

[databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) is another dataset that can be used to train an RLFH model. The approach taken to creating this dataset was quite different from the OpenAssistant Conversations dataset since it was created by employees of Databricks instead of being crowd sourced via volunteers. Perhaps we can use our trained BERTopic model to compare the topics across these two datasets?

The new BERTopic Hub integrations mean we can load this trained model and apply it to new examples. 

```python
topic_model = BERTopic.load("davanstrien/chat_topics")
```

We can predict on a single example text: 

```python
example = "Stalemate is a drawn position. It doesn't matter who has captured more pieces or is in a winning position"
topic, prob = topic_model.transform(example)
```

We can get more information about the predicted topic 

```python
topic_model.get_topic_info(topic)
```

|    |   Count | Name                                  | Representation                                                                                      |
|---:|--------:|:--------------------------------------|:----------------------------------------------------------------------------------------------------|
|  0 |     240 | 22_chess_chessboard_practice_strategy | ['chess', 'chessboard', 'practice', 'strategy', 'learn', 'pawn', 'board', 'pawns', 'play', 'decks'] |

We can see here the topics predicted seems to make sense. We may want to extend this to compare the topics predicted for the whole dataset. 

```python
from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k")
dolly_docs = dataset['train']['response']
dolly_topics, dolly_probs = topic_model.transform(dolly_docs)
```

We can then compare the distribution of topics across both datasets. We cna see here that there seems to be a broader distribution across topics in the dolly dataset according to our BERTopic model. This might be a result of the different approaches to creating both datasets. 

TODO update distribution.png

## Get Started with BERTopic and Hugging Face Hub

You can visit the official documentation for a [quick start guide](https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html) to get help using BERTopic. 

Once you have a trained topic model, you can push it to the Hugging Face Hub in one line. Pushing your model to the Hub will automatically create an initial model card for your model, including an overview of the topics created. You can also find a starter [Colab notebook](https://colab.research.google.com/drive/1JgNfcztTxZP8UEW6qtEyriTcgQfSdzZh?usp=sharing) that shows how you can train a BERTopic model and push it to the Hub.  

Some example of BERTopic models already on the hub:
- [MaartenGr/BERTopic_ArXiv](https://huggingface.co/MaartenGr/BERTopic_ArXiv): a model trained on ~30000 ArXiv Computation and Language articles (cs.CL) after 1991.
- [https://huggingface.co/MaartenGr/BERTopic_Wikipedia](https://huggingface.co/MaartenGr/BERTopic_Wikipedia): a model trained on 1000000 English Wikipedia pages.
- [davanstrien/imdb_bertopic](https://huggingface.co/davanstrien/imdb_bertopic): a model trained on the unsupervised split of the IMDB dataset

We invite you to explore the possibilities of this new integration and share your trained models on the hub!



