---
title: 'Embedding-as-a-Service using SentenceTransformers, the 🤗Inference API, and 🤗Datasets'
thumbnail: /blog/assets/80_ST_inference-api/thumbnail.png
---

<h1>
    Embedding-as-a-Service using SentenceTransformers, the 🤗Inference API, and 🤗Datasets
</h1>

<div class="blog-metadata">
    <small>Published June 13, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/st-inference-api.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/espejelomar"> 
        <img class="avatar avatar-user" src="https://bafybeidj6oxo7zm5pejnc2iezy24npw4qbt2jgpo4n6igt7oykc7rbvcxi.ipfs.dweb.link/omar_picture.png" title="Gravatar">
        <div class="bfc">
            <code>espejelomar</code>
            <span class="fullname">Omar Espejel</span>
        </div>
    </a>
</div>

## Embeddings as a nerve center for industrial applications

Embeddings are essential for modern machine learning. For references, see Pinecone's "[What are Vector Embeddings](https://www.pinecone.io/learn/vector-embeddings/)" and our 101 on [how to use embeddings for semantic search](https://huggingface.co/spaces/sentence-transformers/Sentence_Transformers_for_semantic_search). For generating embeddings, we will be using the open-source library called [SentenceTransformers](https://www.sbert.net/index.html) (ST). 

> "ST is a Python framework for state-of-the-art sentence, text, and image embeddings. [...] You can use ST to compute sentence/text embeddings for more than 100 languages. These embeddings can then be compared, e.g., with cosine-similarity to find sentences with similar meanings. This can be useful for semantic textual similarity, semantic search, or paraphrase mining." - [ST documentation](https://www.sbert.net/index.html#sentencetransformers-documentation).

Once a piece of information (a sentence, a document, an image) is embedded, the creativity starts; several interesting industrial applications use embeddings. E.g., Google Search uses embeddings to [match text to text and text to images](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings); Snapchat uses them to "[serve the right ad to the right user at the right time](https://eng.snap.com/machine-learning-snap-ad-ranking)"; and Meta (Facebook) uses it for [their social search](https://research.facebook.com/publications/embedding-based-retrieval-in-facebook-search/).

> "[...] once you understand this ML multitool (embedding), you'll be able to build everything from search engines to recommendation systems to chatbots and a whole lot more. You don't have to be a data scientist with ML expertise to use them, nor do you need a huge labeled dataset." - [Dale Markowitz, Google Cloud](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings).

But first, we need to embed our dataset (we'll use the terms encode and embed interchangeably). The Hugging Face Inference API allows us to embed a dataset using a quick POST call easily.

In this post, we will:
1. Embed text examples using the 🤗Inference-API.
2. Upload our embedded dataset to the 🤗Hub for free hosting.

## Using the 🤗Inference API to embed our dataset

First, select the model you will use. We can choose a model from the [SentenceTransformers library](https://huggingface.co/sentence-transformers). In this case, we use ["sentence-transformers/all-MiniLM-L6-v2"](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). In the Python code below, we will store the model id in the `model_id` variable.

Log in to the Hugging Face Hub. You must create a token in your [Account Settings](http://hf.co/settings/tokens). We will store your token in `hf_token`.

```py
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "get your token in http://hf.co/settings/tokens"
```
To generate our embeddings we use the `https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}` endpoint with the headers `{"Authorization": f"Bearer {hf_token}"}`. We define a function that would receive a dictionary with our texts and return a list with our embeddings.

```py
from typing import Dict

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts: Dict):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()
```

For the sake of the example, we will be embedding only three texts. The current API does not enforce strict rate limitations. Instead, Hugging Face balances the loads evenly between all our available resources and favors steady flows of requests. If you need to embed several texts or images, the [🤗 Accelerated Inference API](https://huggingface.co/docs/api-inference/index) would speed the inference and let you choose between using a CPU or GPU. 

```py
texts = ["First text", "Hi!", "I love SentenceTransformers"]

output = query(dict(inputs = texts))
```
We get back a list with the three embeddings. Our model, ["sentence-transformers/all-MiniLM-L6-v2"](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), is encoding the three input texts to three embeddings of size 384 each. Let's convert the list to a NumPy array of shape (3,384).

```py
import numpy as np
embeddings = np.asarray(output)

print(embeddings)
```
```py
[[-0.01706981  0.05439235  0.00906055 ...  0.09818567  0.10090933  -0.0086112 ]
 [-0.08946919  0.03267909  0.01558172 ...  0.0648054  -0.01970844   0.0207535 ]
 [-0.09226393 -0.01157228  0.00867325 ...  0.10857432  0.05021105  -0.06054248]]
```

## Host your embedded dataset for free on the 🤗Hub

🤗Datasets is a library for quickly accessing and sharing datasets. We will use it to host our embeddings dataset in the 🤗Hub. Then we can load it with a single line of code. The Datasets documentation is exceptionally helpful on the steps to share and load.; here, we based ourselves on the [Share doc](https://huggingface.co/docs/datasets/share). 

First, login into the 🤗Hub from your terminal. Use a token from your [Account Settings](http://hf.co/settings/tokens).

```
huggingface-cli login
```
Create a dataset repository. You can add an organization.

```
huggingface-cli repo create my_embeddings --type dataset --organization the_embedding_org
```
Clone your repository. Make sure you have [git-lfs installed](https://git-lfs.github.com/). Here the namespace is either your username or your organization name.

```
git lfs install

git clone https://huggingface.co/datasets/namespace/my_embeddings
```

Now let's host our embedding. See the 🤗Datasets documentation for more information on [how to prepare your files](https://huggingface.co/docs/datasets/share#prepare-your-files). First, we add the large data files with `git lfs track`. Suppose our embeddings dataset is compressed in a GZ format.

```
cp /somewhere/data/*.gz .
git lfs track *.gz
git add .gitattributes
git add *.gz
git commit -m "add gz files"
```
Commit and push. You can also upload a loading script, a metadata file, and more. Refer to the [documentation](https://huggingface.co/docs/datasets/share#upload-your-files).

```
git status
git commit -m "First version of the my_embeddings dataset."
git push
```

Now your dataset is hosted on the 🤗Hub for free. You (or whoever you want to share the embeddings with) can quickly load them from a terminal with `embeddings = load_dataset("namespace/my_embeddings")`. 
