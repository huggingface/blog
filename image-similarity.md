---
title: Building an Image Similarity System with 🤗 Datasets and Transformers
thumbnail: /blog/assets/90_tf_serving_vision/thumbnail.png
---

<h1>
  Building an Image Similarity System with 🤗 Datasets and Transformers
</h1>

<div class="blog-metadata">
    <small>Published December 59, 3047.</small> <!-- TODO -->
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/image-similarity.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="https://hf.co/sayakpaul">
        <img class="avatar avatar-user" src="https://avatars.githubusercontent.com/u/22957388?v=4" title="Gravatar">
        <div class="bfc">
            <code>sayakpaul</code>
            <span class="fullname">Sayak Paul</span>
            <span class="bg-gray-100 dark:bg-gray-700 rounded px-1 text-gray-600 text-sm font-mono">guest</span>
        </div>
    </a>
</div>

<a target="_blank" href="https://colab.research.google.com/github/sayakpaul/notebooks/blob/feat/image-sim/examples/image_similarity.ipynb"> <!-- TODO -->
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

In this post, you'll learn to build an image similarity system with 🤗 Transformers. Finding out the similarity between a query image and potential candidates is an important use case for information retrieval systems, reverse image search, for example. All the system is trying to answer is that, given a _query_ image and a set of _candidate_ images, which images are the most similar to the query image. 

We'll leverage the [🤗 `datasets` library](https://huggingface.co/docs/datasets/) as it seamlessly supports parallel processing which will come in handy when building this system. 

Although the post uses a ViT-based model ([`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans)) and a particular dataset ([Beans](https://huggingface.co/datasets/beans)), it can be easily extended to use other models supporting vision modality and other image datasets. Some notable models, you could try:

* [Swin Transformer](https://huggingface.co/docs/transformers/model_doc/swin)
* [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext)
* [RegNet](https://huggingface.co/docs/transformers/model_doc/regnet)

Also, the approach presented in the post can potentially be extended to other modalities as well.

## How do we define similarity?

To build this system, we first need to define how we want to compute the similarity between two images. One widely popular practice is to compute dense representations (embeddings) of the given images and then use the [cosine similarity metric](https://en.wikipedia.org/wiki/Cosine_similarity) to determine how similar the two images are. 

For this post, we'll be using embeddings as a means to represent images in vector space. This gives us a nice way to meaningfully compress the high-dimensional pixel space of images (224 x 224 x 3, for example) to something much lower dimensional (2048, for example). The primary advantage in doing this is the reduced subsequent computations. 

Don't worry if these things do not make sense at all. We'll discuss these things in more detail shortly. 

## Computing embeddings

To compute the embeddings from the images, we'll use a vision model that has some understanding of how to represent the input images in the vector space. This type of models is also commonly referred to as image encoders.

For loading the model, we leverage the [`AutoModel` class](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel). It provides an interface for us to load any compatible model checkpoint from the 🤗 Hub. Alongside the model, we also load the processor associated with the model for data preprocessing. 

```py
from transformers import AutoFeatureExtractor, AutoModel


model_ckpt = "nateraw/vit-base-beans"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
```

In this case, the checkpoint was obtained by fine-tuning a [Vision Transformer based model](https://huggingface.co/google/vit-base-patch16-224-in21k) on the [`beans` dataset](https://huggingface.co/datasets/beans).

Some questions that might arise here:

**Q1**: Why did we not use `AutoModelForImageClassification`?

This is because we want to obtain dense representations of the images and not discrete categories, which are what `AutoModelForImageClassification` would have provided.

**Q2**: Why this checkpoint in particular?

We're using a specific dataset to build the system as mentioned earlier. So, instead of using a generalist model (like the [ones trained on the ImageNet-1k dataset](https://huggingface.co/models?dataset=dataset:imagenet-1k&sort=downloads), for example), it's better to use a model that has been fine-tuned on the dataset being used. That way, the underlying model has a better understanding of the input images.

**Note** that you can also use a checkpoint that was obtained through self-supervised pre-training. The checkpoint doesn't necessarily have to come from supervised learning. In fact, if pre-trained well, self-supervised models can [yield](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/) impressive retrieval
performance. 

Now that we have a model for computing the embeddings, we need some candidate images to query against. 

## Loading the dataset for candidate images

In some time, we'll be building hash tables mapping the candidate images to hashes. During the query time, we'll use these hash tables. We'll talk more about hash tables in the respective section but for now, to have a set of candidate images, we will use the `train` split of the [`beans` dataset](https://huggingface.co/datasets/beans). 

```py
from datasets import load_dataset


dataset = load_dataset("beans")
```

This is how a single sample from the training split looks like:

<div align="center">
    <img src=https://i.ibb.co/f9ycGzw/image.png/ width=600/> <!-- TODO -->
</div>

The dataset has got three columns / features:

```bash
{'image_file_path': Value(dtype='string', id=None),
 'image': Image(decode=True, id=None),
 'labels': ClassLabel(names=['angular_leaf_spot', 'bean_rust', 'healthy'], id=None)}
```

Next, we implement the hashing utilities to optimize the runtime of our image similarity system.

## Random projection and locality-sensitive hashing (LSH)

We can choose to just compute the embeddings with our base model and then apply a similarity metric for the system. But in realistic settings, the embeddings are still high dimensional (in this case `(768, )`). This eats up storage and also increases the query time. 

To this end, we implement the following things:

* First, we reduce the dimensionality of the embeddings with [random projection](https://cs-people.bu.edu/evimaria/cs565/kdd-rp.pdf). The main idea is that if the distance between a group of vectors can roughly be preserved on a plane, the dimensionality of the plane can be further reduced. 
* We then compute the bitwise hash values of the projected vectors to determine their hash buckets. Similar images will likely be closer in the embedding space. Therefore, they will likely also have the same hash values and are likely to go into the same hash bucket. From a deployment perspective, bitwise hash values are cheaper to store and operate on. If you're unfamiliar with the relevant concepts of hashing, then [this resource](https://computersciencewiki.org/index.php/Hashing.) could be helpful. 


```py
# Define random vectors to project with.
RANDOM_VECS = np.random.randn(hash_size, hidden_dim).T


def hash_func(embedding, random_vectors=RANDOM_VECS):
    """Randomly projects the embeddings and then computes bit-wise hashes."""
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    if len(embedding.shape) < 2:
        embedding = np.expand_dims(embedding, 0)

    # Random projection.
    bools = np.dot(embedding, random_vectors) > 0
    return [bool2int(bool_vec) for bool_vec in bools]


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        if j:
            y += 1 << i
    return y
```

Next, we define a utility that can be mapped to our dataset for computing hashes of the training images in a parallel manner. 

```py
def compute_hash(model: torch.nn.Module):
    """Computes hash on a given dataset."""
    device = model.device

    def pp(example_batch):
        # Prepare the input images for the model.
        image_batch = example_batch["image"]
        image_batch_transformed = torch.stack(
            [transformation_chain(image) for image in image_batch]
        )
        new_batch = {
            "pixel_values": image_batch_transformed.to(device)
            if isinstance(model, torch.nn.Module)
            else image_batch_transformed.numpy()
        }

        # Compute embeddings and pool them i.e., take the representations from the [CLS]
        # token.
        with torch.no_grad():
            embeddings = model(**new_batch).last_hidden_state[:, 0].cpu().numpy()
            
        # Compute hashes for the batch of images.
        hashes = [hash_func(embeddings[i]) for i in range(len(embeddings))]
        example_batch["hashes"] = hashes
        return example_batch

    return pp
```

Next, we build three utility classes building our hash tables:

* `Table`
* `LSH`
* `BuildLSHTable` 

Collectively, these classes implement Locality Sensitive Hashing (the idea locally close points share the same hashes). 

**Disclaimer**: Some code has been used from [this resource](https://keras.io/examples/vision/near_dup_search/) for writing these classes. 

## The `Table` class

The `Table` class has two methods:

* `add()` lets us build a dictionary mapping the hashes of the candidate images to their identifiers. 
* `query()` lets us take as inputs the query hashes and check if they exist in the table.

The table built in this class is referred to as a hash bucket. 

```py
from typing import List


class Table:
    def __init__(self, hash_size: int):
        self.table = {}
        self.hash_size = hash_size

    def add(self, id: int, hashes: List[int], label: int):
        # Create a unique indentifier.
        entry = {"id_label": str(id) + "_" + str(label)}

        # Add the hash values to the current table.
        for h in hashes:
            if h in self.table:
                self.table[h].append(entry)
            else:
                self.table[h] = [entry]

    def query(self, hashes: List[int]):
        results = []

        # Loop over the query hashes and determine if they exist in
        # the current table.
        for h in hashes:
            if h in self.table:
                results.extend(self.table[h])
        return results
```

## The `LSH` class 

Our dimensionality reduction technique involves a degree of randomness. This can lead to a situation where similar images may not get mapped to the same hash bucket every time the process is run. To reduce this effect, we'll maintain multiple hash tables. The number of hash tables and the reduction dimensionality are the two key hyperparameters here. 

```py
class LSH:
    def __init__(self, hash_size, num_tables):
        self.num_tables = num_tables
        self.tables = []
        for i in range(self.num_tables):
            self.tables.append(Table(hash_size))

    def add(self, id: int, hash: List[int], label: int):
        for table in self.tables:
            table.add(id, hash, label)

    def query(self, hashes: List[int]):
        results = []
        for table in self.tables:
            results.extend(table.query(hashes))
        return results
```

## The `BuildLSHTable` class

It lets us:

* `build()`: build the hash tables. 
* `query()` with an input image aka the query image. 

```py
from tqdm.auto import tqdm
from PIL import Image
import datasets


class BuildLSHTable:
    def __init__(
        self,
        model: Union[torch.nn.Module, str],
        batch_size: int = 48,
        hash_size: int = hash_size,
        dim: int = hidden_dim,
        num_tables: int = 10,
    ):
        self.hash_size = hash_size
        self.dim = dim
        self.num_tables = num_tables
        self.lsh = LSH(self.hash_size, self.num_tables)

        self.batch_size = batch_size
        self.model = model
        self.hash_fn = compute_hash(
            self.model.to(device)
            if isinstance(self.model, torch.nn.Module)
            else self.model
        )

    def build(self, ds: datasets.DatasetDict):
        dataset_hashed = ds.map(self.hash_fn, batched=True, batch_size=self.batch_size)

        for id in tqdm(range(len(dataset_hashed))):
            hash, label = dataset_hashed[id]["hashes"], dataset_hashed[id]["labels"]
            self.lsh.add(id, hash, label)

    def query(self, image, verbose=True):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Compute the hashes of the query image and fetch the results.
        example_batch = dict(image=[image])
        hashes = self.hash_fn(example_batch)["hashes"][0]

        results = self.lsh.query(hashes)
        if verbose:
            print("Matches:", len(results))

        # Calculate Jaccard index to quantify the similarity.
        counts = {}
        for r in results:
            if r["id_label"] in counts:
                counts[r["id_label"]] += 1
            else:
                counts[r["id_label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.dim
        return counts
```

**Notes on quantifying similarity**:

We're using [Jaccard index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html) to quantify the similarity between the query image and the candidate images. As per [Scikit Learn's documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html), "it is defined as the size of the intersection divided by the size of the union of two label sets."

Since we're using LSH to build the similarity system and the hashes are effectively sets, Jaccard index is a good metric to use here. 

## Building the LSH tables

With the above utilities, we're ready to build the LSH tables.

```py
lsh_builder = BuildLSHTable(model)
lsh_builder.build(dataset["train"])
```

To get a better a idea of how the tables are represented internally within `lsh_builder`, let's investigate the contents of a single table.

```py
idx = 0
for hash, entry in lsh_builder.lsh.tables[0].table.items():
    if idx == 5:
        break
    if len(entry) < 5:
        print(f"Hash: {hash}, entries: {entry}")
        idx += 1
```

You should get something similar to:

```bash
Hash: 57, entries: [{'id_label': '16_0'}, {'id_label': '213_0'}, {'id_label': '632_1'}]
Hash: 24, entries: [{'id_label': '27_0'}]
Hash: 136, entries: [{'id_label': '31_0'}, {'id_label': '168_0'}, {'id_label': '286_0'}, {'id_label': '340_0'}]
Hash: 193, entries: [{'id_label': '45_0'}, {'id_label': '47_0'}, {'id_label': '94_0'}, {'id_label': '231_0'}]
Hash: 185, entries: [{'id_label': '63_0'}, {'id_label': '254_0'}]
```

We notice that for a given hash value, we have entries where labels are the same. Because of the randomness induced in the process, we may also notice some entries coming from different labels. It can happen for various reasons:

* The reduction dimensionality is too small for compression. 
* The underlying images may be visually quite similar to one another yet have different labels. 

In both of the above cases, experimentation is really the key to improving the results. 

Now that the LSH tables have been built, we can use them to query them with images. 

## Inference

In this secton, we'll take query images from the `test` split of our dataset and retrieve the similar images from the set of candidate images we have. 

```py
def visualize_lsh(lsh_class: BuildLSHTable, top_k: int = 5):
    idx = np.random.choice(len(dataset["test"]))

    image = dataset["test"][idx]["image"]
    label = dataset["test"][idx]["labels"]
    results = lsh_class.query(image)

    candidates = []
    labels = []
    overlaps = []

    for idx, r in enumerate(sorted(results, key=results.get, reverse=True)):
        if idx == (top_k - 1):
            break
        image_id, label = r.split("_")[0], r.split("_")[1]
        candidates.append(dataset["train"][int(image_id)]["image"])
        labels.append(label)
        overlaps.append(results[r])

    candidates.insert(0, image)
    labels.insert(0, label)

    plot_images(candidates, labels)
```

Let's now put `visualize_lsh()` to use:

```py
visualize_lsh(lsh_builder)
```

<div align="center">
    <img src="https://i.ibb.co/L5rsj3r/image.png"/> <!-- TODO -->
</div>

## Conclusion

TODO