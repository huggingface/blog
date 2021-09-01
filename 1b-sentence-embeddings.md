---
title: "Train a Sentence Embedding Model with 1B Training Pairs"
thumbnail: /blog/assets/22_gradio/gradio.png
---

<h1>
    Train a Sentence Embedding Model with 1B Training Pairs
</h1>

<div class="blog-metadata">
    <small>Published June 28, 2021.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/master/1b-sentence-embeddings.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/asi">
        <img class="avatar avatar-user" src="https://twitter.com/antoinesimoulin/photo" title="Gravatar">
        <div class="bfc">
            <code>asi</code>
            <span class="fullname">Antoine Simoulin</span>
        </div>
    </a>
</div>

*Sentence embedding* is a method that maps sentences to vectors of real numbers. Ideally, these vectors would capture the semantic of a sentence and be highly generic. Such representations could then be used for many downstream applications such as clustering, text mining, or question answering.

We developed state-of-the-art sentence embedding models as part of the project [Train the Best Sentence Embedding Model Ever with 1B Training Pairs](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354). This project took place during the [Community week using JAX/Flax for NLP & CV](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104), organized by Hugging Face.  We benefited from efficient hardware infrastructure to run the project: 7 TPUs v3-8, as well as intervention from Google’s Flax, JAX, and Cloud team members about efficient deep learning frameworks!

## Training methodology

### Model

Unlike words, we can not define a finite set of sentences. Sentence embedding methods, therefore, compose inner words to compute the final representation. For example, the SentenceBert ([Reimers and Gurevych, 2019](https://aclanthology.org/D19-1410.pdf)) model uses Transformer, the cornerstone of many NLP applications.

![snippet](assets/25_1b_sentence_embeddings/model.png)

### Multiple Negative Ranking Loss

The parameters from the composition module are usually learned using a self-supervised objective. We used a contrastive training method: we constitute a dataset with sentence pairs (a_i, p_i) such that sentences from the pair have a close meaning. For example, we consider pairs such as (query, answer-passage), (question, duplicate_question),(paper title, cited paper title). Our model is then trained to map pairs $(a_i , p_i)$ to close vectors while assigning unmatched pairs $(a_i , p_j), i !=j$ to distant vectors in the embedding space. This training method is also called training with in-batch negatives, InfoNCE or NTXentLoss.

![snippet](assets/25_1b_sentence_embeddings/contrastive_1.png)

Given a batch of training samples, the model optimises the following [loss function](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py):

\(-\frac{1}{n}\sum_{i=1}^n\frac{exp(sim(a_i, p_i))}{\sum_j exp(sim(a_i, p_j))}\)

To give you a sense about the loss, we have illustrated an example below. Intruitively, the model should assign high similarity to the sentences « How many people live in Berlin? » and « Around 3.5 million people live in Berlin » and low similarity to other negative answers such as « The capital of France is Paris » as detailed in the Figure.

![snippet](assets/25_1b_sentence_embeddings/contrastive_2.png)

In the loss equation, Sim indicates a similarity function between (a, p). The similarity function could be either the Cosine-Similarity or the Dot-Product operator. Both methods have their pros and cons summarized below ([Thakur et al., 2021](https://arxiv.org/abs/2104.08663), [Bachrach et al., 2014](https://dl.acm.org/doi/10.1145/2645710.2645741))):

| Cosine-similarity.  | Dot-product |
|————————————-------——|——---————————|
| Vector has highest similarity to itself since $cos(a, a)=1$  |  Other vectors can have higher dot-products $dot(a, a) < dot (a, b)$ |
| With normalised vectors it is equal to the dot product. The max vector length is equals 1  | It might be slower with certain approximate nearest neighbour methods since the max vector not known |
| With normalised vectors, it is proportional to euclidian distance. It works with k-means clustering  | It does not work with k-means clustering  |

In practise, we used a scaled similarity because score differences tends to be to small and apply a scaling factor $C$ such that $sim_scaled(a, b) = C * sim(a, b) $ with typically $C = 20$ ([Henderson et al., 2020]([https://doi.org/10.18653/v1/2020.findings-emnlp.196), [Radford et al., 2021](http://proceedings.mlr.press/v139/radford21a.html)).

### Improving Quality with Better Batches

In our method, we build batches of sample pairs $(a_i , p_i)$. We consider all other samples from the batch, $(a_i , p_j), i != j$, as negatives sample pairs. The batch composition is therefore a key training aspect. Given the literature in the domain, we particularly focused on three main aspects of the batch.

#### Size matters

In contrastive learning Larger batch size is synonymous with better performance. As shown in the Figure extracted from Qu and al., [2021](https://doi.org/10.18653/v1/2021.naacl-main.466), a larger batch size increases the results.[image:4902FEC1-A32F-4DFD-8D15-054EBA5F9C78-442-000201046959DC02/F67900A8-8936-459B-AAAA-213C4CE45340.png]

#### Hard Negatives

In the same figure, we observe that including hard negatives also improves performance. Hard negatives are sample $p_j$ which are hard to distinguish from $p_i$. In our example, it could be the pairs « What is the capital of France? » and « What is the capital of the US? » which have a close semantic content and requires precisely understand the full sentence to be answered. On the contrary, the samples  « What is the capital of France? » and «How many Star Wars movies is there?» are less difficult to distinguish since they do not refer to the same topic.

#### Cross dataset batches

We concatenated multiple datasets to train our models. We built a large batch and gather samples from the same dataset in the batch in order to limit the topic distribution and favor hard negatives. However, we also mix at least two datasets in the batch, to learn a global structure between topics and not only a local structure within a topic.

## Training infrastructure and data

As mention earlier, the quantity of data and the batch size directly impact the model performances. As part of the project, we benefited from efficient hardware infrastructure. We trained our models on [TPUs](https://cloud.google.com/tpu) which are compute units developed by Google and super efficient for matrix multiplications. TPUs have some [hardware specificities](https://huggingface.co/docs/accelerate/quicktour.html#training-on-tpu) which might require some specific code implementation.

Additionally, we trained models on a large corpus as we concatenated multiple datasets up to 1 billion sentence pairs! All datasets used are detailed for each model in the [model card](https://huggingface.co/flax-sentence-embeddings/all_datasets_v3_MiniLM-L12).

## Conclusion

You can find all models and datasets we created during the challenge on our [HuggingFace repository](https://huggingface.co/flax-sentence-embeddings). We trained 20 general-purpose Sentence Transformers models such as Mini-LM ([Wang et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)), Roberta ([liu and al., 2019](https://arxiv.org/abs/1907.11692 )), DistilBERT ([Sanh and al., 2020](http://arxiv.org/abs/1910.01108)) or MPNet ([Song and al., 2020](https://proceedings.neurips.cc/paper/2020/hash/c3a690be93aa602ee2dc0ccab5b7b67e-Abstract.html)). Our models Achieve SOTA on multiple general-purpose Sentence Similarity evaluation tasks. We also uploaded  [8 datasets](https://huggingface.co/flax-sentence-embeddings)  specialized for Question Answering, Sentence-Similarity, and Gender Evaluation. 

General sentence embeddings might be used for many applications. We proposed some [demonstration](https://huggingface.co/spaces/flax-sentence-embeddings/sentence-embeddings) for several applications:
* The *sentence similarity* module compares the similarity of the main text with other texts of your choice. In the background, we’ll create an embedding for each text, and then we’ll use the cosine similarity function to calculate a similarity metric between our main sentence and the others.
* *Asymmetric QA* compare the Answer likeliness of a given Query with answer candidates of your choice.
* *Search / Cluster* return nearby answers from a query. For example, if you input « python », it will retrieve anything related in the sense of the dot-product. distance.
* *Gender Bias Evaluation* report *inherent gender bias* in training set via random sampling of the sentences. Given an anchor text without any mention of gender for target occupation and 2 propositions with gendered pronouns, we compare if models assign a higher similarity to a given proposition and therefore evaluate their proportion to favor a specific gender.

The [Community week using JAX/Flax for NLP & CV](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104) has been an intense and highly rewarding experience! The quality of the actors’ interventions and their presence helped us all learn a lot. We hope all projects had as much fun as we did in ours. Whenever you have questions or suggestions, don’t hesitate to contact us!
