---
title: "Large-scale Near-deduplication Behind BigCode"
thumbnail: /blog/assets/dedup/thumbnail.png
authors:
  - user: chenghao
---

# Large-scale Near-deduplication Behind BigCode


## Intended Audience

People who are interested in document-level near-deduplication at a large scale, and have some understanding of hashing, graph and text processing.

## Motivations

It is important to take care of our data before feeding it to the model, at least Large Language Model in our case, as the old saying goes, garbage in, garbage out. Even though it's increasingly difficult to do so with headline-grabbing models (or should we say APIs) creating an illusion that data quality matters less.

One of the problems we face in both BigScience and BigCode for data quality is duplication, including possible benchmark contamination. It has been shown that models tend to output training data verbatim when there are many duplicates[[1]](#1) (though it is less clear in some other domains[[2]](#2)), and it also makes the model vulnerable to privacy attacks[[1]](#1). Additionally, some typical advantages of deduplication also include:

1. Efficient training: You can achieve the same, and sometimes better, performance with less training steps[[3]](#3) [[4]](#4).
2. Prevent possible data leakage and benchmark contamination: Non-zero duplicates discredit your evaluations and potentially make so-called improvement a false claim.
3. Accessibility. Most of us cannot afford to download or transfer thousands of gigabytes of text repeatedly, not to mention training a model with it. Deduplication, for a fix-sized dataset, makes it easier to study, transfer and collaborate with.

## From BigScience to BigCode

Allow me to share a story first on how I jumped on this near-deduplication quest, how the results have progressed, and what lessons I have learned along the way.

It all started with a conversation on LinkedIn when [BigScience](https://bigscience.huggingface.co/) had already started for a couple of months. Huu Nguyen approached me when he noticed my pet project on GitHub, asking me if I were interested in working on deduplication for BigScience. Of course, my answer was a yes, completely ignorant of just how much effort will be required alone due to the sheer mount of the data.

It was fun and challenging at the same time. It is challenging in a sense that I didn't really have much research experience with that sheer scale of data, and everyone was still welcoming and trusting you with thousands of dollars of cloud compute budget. Yes, I had to wake up from my sleep to double-check that I had turned off those machines several times. As a result, I had to learn on the job through all the trials and errors, which in the end opened me to a new perspective that I don't think I would ever have if it weren't for BigScience.

Moving forward, one year later, I am putting what I have learned back into [BigCode](https://www.bigcode-project.org/), working on even bigger datasets. In addition to LLMs that are trained for English[[3]](#3), we have confirmed that deduplication improves code models too[[4]](#4), while using a much smaller dataset. And now, I am sharing what I have learned with you, my dear reader, and hopefully, you can also get a sense of what is happening behind the scene of BigCode through the lens of deduplication.

In case you are interested, here is an updated version of the deduplication comparison that we started in BigScience:

| Dataset                              | Input Size                       | Output Size or Deduction                                        | Level                 | Method                                        | Parameters                                                       | Language     | Time                |
| ------------------------------------ | -------------------------------- | --------------------------------------------------------------- | --------------------- | --------------------------------------------- | ---------------------------------------------------------------- | ------------ | ------------------- |
| OpenWebText2[[5]](#5)                | After URL dedup: 193.89 GB (69M) | After MinHashLSH: 65.86 GB (17M)                                | URL + Document        | URL(Exact) + Document(MinHash LSH)            | \\( (10, 0.5, ?, ?, ?) \\)                                       | English      |                     |
| Pile-CC[[5]](#5)                     | _~306 GB_                        | _227.12 GiB (~55M)_                                             | Document              | Document(MinHash LSH)                         | \\( (10, 0.5, ?, ?, ?) \\)                                       | English      | "several days"      |
| BNE5[[6]](#6)                        | 2TB                              | 570 GB                                                          | Document              | Onion                                         | 5-gram                                                           | Spanish      |                     |
| MassiveText[[7]](#7)                 |                                  | 0.001 TB ~ 2.1 TB                                               | Document              | Document(Exact + MinHash LSH)                 | \\( (?, 0.8, 13, ?, ?) \\)                                       | English      |                     |
| CC100-XL[[8]](#8)                    |                                  | 0.01 GiB ~ 3324.45 GiB                                          | URL + Paragraph       | URL(Exact) + Paragraph(Exact)                 | SHA-1                                                            | Multilingual |                     |
| C4[[3]](#3)                          | 806.92 GB (364M)                 | 3.04% ~ 7.18% **↓** (train)                                     | Substring or Document | Substring(Suffix Array) or Document(MinHash)  | Suffix Array: 50-token, MinHash: \\( (9000, 0.8, 5, 20, 450) \\) | English      |                     |
| Real News[[3]](#3)                   | ~120 GiB                         | 13.63% ~ 19.4% **↓** (train)                                    | Same as **C4**        | Same as **C4**                                | Same as **C4**                                                   | English      |                     |
| LM1B[[3]](#3)                        | ~4.40 GiB (30M)                  | 0.76% ~ 4.86% **↓** (train)                                     | Same as **C4**        | Same as **C4**                                | Same as **C4**                                                   | English      |                     |
| WIKI40B[[3]](#3)                     | ~2.9M                            | 0.39% ~ 2.76% **↓** (train)                                     | Same as **C4**        | Same as **C4**                                | Same as **C4**                                                   | English      |                     |
| The BigScience ROOTS Corpus[[9]](#9) |                                  | 0.07% ~ 2.7% **↓** (document) + 10.61%~32.30% **↓** (substring) | Document + Substring  | Document (SimHash) + Substring (Suffix Array) | SimHash: 6-grams, hamming distance of 4, Suffix Array: 50-token  | Multilingual | 12 hours ~ few days |

This is the one for code datasets we created for BigCode as well. Model names are used when the dataset name isn't available.

| Model                 | Method               | Parameters                             | Level    |
| --------------------- | -------------------- | -------------------------------------- | -------- |
| InCoder[[10]](#10)    | Exact                | Alphanumeric tokens/md5 + Bloom filter | Document |
| CodeGen[[11]](#11)    | Exact                | SHA256                                 | Document |
| AlphaCode[[12]](#12)  | Exact                | ignore whiespaces                      | Document |
| PolyCode[[13]](#13)   | Exact                | SHA256                                 | Document |
| PaLM Coder[[14]](#14) | Levenshtein distance |                                        | Document |
| CodeParrot[[15]](#15) | MinHash + LSH        | \\( (256, 0.8, 1) \\)                  | Document |
| The Stack[[16]](#16)  | MinHash + LSH        | \\( (256, 0.7, 5) \\)                  | Document |

MinHash + LSH parameters \\( (P, T, K, B, R) \\):

1. \\( P \\) number of permutations/hashes
2. \\( T \\) Jaccard similarity threshold
3. \\( K \\) n-gram/shingle size
4. \\( B \\) number of bands
5. \\( R \\) number of rows

To get a sense of how those parameters might impact your results, here is a simple demo to illustrate the computation mathematically: [MinHash Math Demo](https://huggingface.co/spaces/bigcode/near-deduplication).


## MinHash Walkthrough

In this section, we will cover each step of MinHash, the one used in BigCode, and potential scaling issues and solutions. We will demonstrate the workflow via one example of three documents in English:

| doc_id | content                                  |
| ------ | ---------------------------------------- |
| 0      | Deduplication is so much fun!            |
| 1      | Deduplication is so much fun and easy!   |
| 2      | I wish spider dog[[17]](#17) is a thing. |

The typical workflow of MinHash is as follows:

1. Shingling (tokenization) and fingerprinting (MinHashing), where we map each document into a set of hashes.
2. Locality-sensitive hashing (LSH). This step is to reduce the number of comparisons by grouping documents with similar bands together.
3. Duplicate removal. This step is where we decide which duplicated documents to keep or remove.

### Shingles

Like in most applications involving text, we need to begin with tokenization. N-grams, a.k.a. shingles, are often used. In our example, we will be using word-level tri-grams, without any punctuations. We will circle back to how the size of ngrams impacts the performance in a later section.

| doc_id | shingles                                                                        |
| ------ | ------------------------------------------------------------------------------- |
| 0      | {"Deduplication is so", "is so much", "so much fun"}                            |
| 1      | {'so much fun', 'fun and easy', 'Deduplication is so', 'is so much'}            |
| 2      | {'dog is a', 'is a thing', 'wish spider dog', 'spider dog is', 'I wish spider'} |

This operation has a time complexity of \\( \mathcal{O}(NM) \\) where \\( N \\) is the number of documents and \\( M \\) is the length of the document. In other words, it is linearly dependent on the size of the dataset. This step can be easily scaled by parallelization by multiprocessing or distributed computation.

### Fingerprint Computation

In MinHash, each shingle will typically either be 1) hashed multiple times with different hash functions, or 2) permuted multiple times using one hash function. Here, we choose to permute each hash 5 times. More variants of MinHash can be found in [MinHash - Wikipedia](https://en.wikipedia.org/wiki/MinHash?useskin=vector).

| shingle             | permuted hashes                                             |
| ------------------- | ----------------------------------------------------------- |
| Deduplication is so | [403996643, 2764117407, 3550129378, 3548765886, 2353686061] |
| is so much          | [3594692244, 3595617149, 1564558780, 2888962350, 432993166] |
| so much fun         | [1556191985, 840529008, 1008110251, 3095214118, 3194813501] |

Taking the minimum value of each column within each document — the "Min" part of the "MinHash", we arrive at the final MinHash for this document:

| doc_id | minhash                                                    |
| ------ | ---------------------------------------------------------- |
| 0      | [403996643, 840529008, 1008110251, 2888962350, 432993166]  |
| 1      | [403996643, 840529008, 1008110251, 1998729813, 432993166]  |
| 2      | [166417565, 213933364, 1129612544, 1419614622, 1370935710] |

Technically, we don't have to use the minimum value of each column, but the minimum value is the most common choice. Other order statistics such as maximum, kth smallest, or kth largest can be used as well[[21]](#21).

In implementation, you can easily vectorize these steps with `numpy` and expect to have a time complexity of \\( \mathcal{O}(NMK) \\) where \\( K \\) is your number of permutations. Code modified based on [Datasketch](https://github.com/ekzhu/datasketch).

```python
def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> Dict[str, Any]:
    a, b = permutations
    masks: np.ndarray = np.full(shape=num_perm, dtype=np.uint64, fill_value=MAX_HASH)
    tokens: Set[str] = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    hashvalues: np.ndarray = np.array([sha1_hash(token.encode("utf-8")) for token in tokens], dtype=np.uint64)
    permuted_hashvalues = np.bitwise_and(
        ((hashvalues * np.tile(a, (len(hashvalues), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )
    hashvalues = np.vstack([permuted_hashvalues, masks]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}
```

If you are familiar with [Datasketch](https://github.com/ekzhu/datasketch), you might ask, why do we bother to strip all the nice high-level functions the library provides? It is not because we want to avoid adding dependencies, but because we intend to squeeze as much CPU computation as possible during parallelization. Fusing few steps into one function call enables us to utilize our compute resources better.

Since one document's calculation is not dependent on anything else. A good parallelization choice would be using the `map` function from the `datasets` library:

```python
embedded = ds.map(
	function=embed_func,
	fn_kwargs={
		"num_perm": args.num_perm,
		"hashranges": HASH_RANGES,
		"ngram_size": args.ngram,
		"permutations": PERMUTATIONS,
	},
	input_columns=[args.column],
	remove_columns=ds.column_names,
	num_proc=os.cpu_count(),
	with_indices=True,
	desc="Fingerprinting...",
)
```

After the fingerprint calculation, one particular document is mapped to one array of integer values. To figure out what documents are similar to each other, we need to group them based on such fingerprints. Entering the stage, **Locality Sensitive Hashing (LSH)**.

### Locality Sensitive Hashing

LSH breaks the fingerprint array into bands, each band containing the same number of rows. If there is any hash values left, it will be ignored. Let's use \\( b=2 \\) bands and \\( r=2 \\) rows to group those documents:

| doc_id | minhash                                                    | bands                                                  |
| ------ | ---------------------------------------------------------- | ------------------------------------------------------ |
| 0      | [403996643, 840529008, 1008110251, 2888962350, 432993166]  | [0:[403996643, 840529008], 1:[1008110251, 2888962350]] |
| 1      | [403996643, 840529008, 1008110251, 1998729813, 432993166]  | [0:[403996643, 840529008], 1:[1008110251, 1998729813]] |
| 2      | [166417565, 213933364, 1129612544, 1419614622, 1370935710] | [0:[166417565, 213933364], 1:[1129612544, 1419614622]] |

If two documents share the same hashes in a band at a particular location (band index), they will be clustered into the same bucket and will be considered as candidates.

| band index | band value               | doc_ids |
| ---------- | ------------------------ | ------- |
| 0          | [403996643, 840529008]   | 0, 1    |
| 1          | [1008110251, 2888962350] | 0       |
| 1          | [1008110251, 1998729813] | 1       |
| 0          | [166417565, 213933364]   | 2       |
| 1          | [1129612544, 1419614622] | 2       |

For each row in the `doc_ids` column, we can generate candidate pairs by pairing every two of them. From the above table, we can generate one candidate pair: `(0, 1)`.

### Beyond Duplicate Pairs

This is where many deduplication descriptions in papers or tutorials stop. We are still left with the question of what to do with them. Generally, we can proceed with two options:

1. Double-check their actual Jaccard similarities by calculating their shingle overlap, due to the estimation nature of MinHash. The Jaccard Similarity of two sets is defined as the size of the intersection divided by the size of the union. And now it becomes much more doable than computing all-pair similarities, because we can focus only for documents within a cluster. This is also what we initially did for BigCode, which worked reasonably well.
2. Treat them as true positives. You probably already noticed the issue here: the Jaccard similarity isn't transitive, meaning \\( A \\) is similar to \\( B \\) and \\( B \\) is similar to \\( C \\), but \\( A \\) and \\( C \\) do not necessary share the similarity. However, our experiments from The Stack show that treating all of them as duplicates improves the downstream model's performance the best. And now we gradually moved towards this method instead, and it saves time as well. But to apply this to your dataset, we still recommend going over your dataset and looking at your duplicates, and then making a data-driven decision.

From such pairs, whether they are validated or not, we can now construct a graph with those pairs as edges, and duplicates will be clustered into communities or connected components. In terms of implementation, unfortunately, this is where `datasets` couldn't help much because now we need something like a `groupby` where we can cluster documents based on their _band offset_ and _band values_. Here are some options we have tried:

**Option 1: Iterate the datasets the old-fashioned way and collect edges. Then use a graph library to do community detection or connected component detection.**

This did not scale well in our tests, and the reasons are multifold. First, iterating the whole dataset is slow and memory consuming at a large scale. Second, popular graph libraries like `graphtool` or `networkx` have a lot of overhead for graph creation.

**Option 2: Use popular python frameworks such as `dask` to allow more efficient `groupby` operations**.

But then you still have problems of slow iteration and slow graph creation.

**Option 3: Iterate the dataset, but use a union find data structure to cluster documents.**

This adds negligible overhead to the iteration, and it works relatively well for medium datasets.

```python
for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
	for cluster in table.values():
		if len(cluster) <= 1:
			continue
		idx = min(cluster)
		for x in cluster:
			uf.union(x, idx)
```

**Option 4: For large datasets, use Spark.**

We already know that steps up to the LSH part can be parallelized, which is also achievable in Spark. In addition to that, Spark supports distributed `groupBy` out of the box, and it is also straightforward to implement algorithms like [[18]](#18) for connected component detection. If you are wondering why we didn't use Spark's implementation of MinHash, the answer is that all our experiments so far stemmed from [Datasketch](https://github.com/ekzhu/datasketch), which uses an entirely different implementation than Spark, and we want to ensure that we carry on the lessons and insights learned from that without going into another rabbit hole of ablation experiments.

```python
edges = (
	records.flatMap(
		lambda x: generate_hash_values(
			content=x[1],
			idx=x[0],
			num_perm=args.num_perm,
			ngram_size=args.ngram_size,
			hashranges=HASH_RANGES,
			permutations=PERMUTATIONS,
		)
	)
	.groupBy(lambda x: (x[0], x[1]))
	.flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
	.distinct()
	.cache()
)
```

A simple connected component algorithm based on [[18]](#18) implemented in Spark.

```python
a = edges
while True:
	b = a.flatMap(large_star_map).groupByKey().flatMap(large_star_reduce).distinct().cache()
	a = b.map(small_star_map).groupByKey().flatMap(small_star_reduce).distinct().cache()
	changes = a.subtract(b).union(b.subtract(a)).collect()
	if len(changes) == 0:
		break

results = a.collect()
```

Additionally, thanks to cloud providers, we can set up Spark clusters like a breeze with services like GCP DataProc. **In the end, we can comfortably run the program to deduplicate 1.4 TB of data in just under 4 hours with a budget of $15 an hour.**

## Quality Matters

Scaling a ladder doesn't get us to the moon. That's why we need to make sure this is the right direction, and we are using it the right way.

Early on, our parameters were largely inherited from the CodeParrot experiments, and our ablation experiment indicated that those settings did improve the model's downstream performance[[16]](#16). We then set to further explore this path and can confirm that[[4]](#4):

1. Near-deduplication improves the model's downstream performance with a much smaller dataset (6 TB VS. 3 TB)
2. We haven't figured out the limit yet, but a more aggressive deduplication (6 TB VS. 2.4 TB) can improve the performance even more:
   1. Lower the similarity threshold
   2. Increase the shingle size (unigram → 5-gram)
   3. Ditch false positive checking because we can afford to lose a small percentage of false positives

![A violin chart showing unigram impact in different settings](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/violin_chart_1.png)
![A violin chart showing unigram impact in different settings](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/violin_chart_2.png)

<center>
Image: Two graphs showing the impact of similarity threshold and shingle size, the first one is using unigram and the second one 5-gram. The red dash line shows the similarity cutoff: any documents below would be considered as false positives — their similarities with other documents within a cluster are lower than the threshold.
</center>

These graphs can help us understand why it was necessary to double-check the false positives for CodeParrot and early version of the Stack: using unigram creates many false positives; They also demonstrate that by increasing the shingle size to 5-gram, the percentage of false positives decreases significantly. A smaller threshold is desired if we want to keep the deduplication aggressiveness.

Additional experiments also showed that lowering the threshold removes more documents that have high similarity pairs, meaning an increased recall in the segment we actually would like to remove the most.

## Scaling

![Scaling results for dataset size and deduplication time](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/scale.png)

<center>Image: Deduplication time versus raw dataset size. This is achieved with 15 worker c2d-standard-16 machines on GCP, and each costed around $0.7 per hour. </center>

![CPU usage screenshot for the cluster during processing JSON dataset](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/usage.png)

<center>Image: CPU usage screenshot for the cluster during processing JSON dataset.</center>

This isn't the most rigorous scaling proof you can find, but the deduplication time, given a fixed computation budget, looks practically linear to the physical size of the dataset. When you take a closer look at the cluster resource usage when processing JSON dataset, the largest subset in the Stack, you can see the MinHash + LSH (stage 2) dominated the total real computation time (stage 2 + 3), which from our previous analysis is \\( \mathcal{O}(NM) \\) — linear to the dataset physical volume.

## Proceed with Caution

Deduplication doesn't exempt you from thorough data exploration and analysis. In addition, these deduplication discoveries hold true for the Stack, but it does not mean it is readily applicable to other datasets or languages. It is a good first step towards building a better dataset, and further investigations such as data quality filtering (e.g., vulnerability, toxicity, bias, generated templates, PII) are still much needed. 

We still encourage you to perform similar analysis on your datasets before training. For example, it might not be very helpful to do deduplication if you have tight time and compute budget: [@geiping_2022](http://arxiv.org/abs/2212.14034) mentions that substring deduplication didn't improve their model's downstream performance. Existing datasets might also require thorough examination before use, for example, [@gao_2020](http://arxiv.org/abs/2101.00027) states that they only made sure the Pile itself, along with its splits, are deduplicated, and they won't proactively deduplicating for any downstream benchmarks and leave that decision to readers.

In terms of data leakage and benchmark contamination, there is still much to explore. We had to retrain our code models because HumanEval was published in one of the GitHub repos in Python. Early near-deduplication results also suggest that MBPP[[19]](#19), one of the most popular benchmarks for coding, shares a lot of similarity with many Leetcode problems (e.g., task 601 in MBPP is basically Leetcode 646, task 604 ≃ Leetcode 151.). And we all know GitHub is no short of those coding challenges and solutions. It will be even more difficult if someone with bad intentions upload all the benchmarks in the form of python scripts, or other less obvious ways, and pollute all your training data.

## Future Directions

1. Substring deduplication. Even though it showed some benefits for English[[3]](#3), it is not clear yet if this should be applied to code data as well;
2. Repetition: paragraphs that are repeated multiple times in one document. [@rae_2021](http://arxiv.org/abs/2112.11446) shared some interesting heuristics on how to detect and remove them.
3. Using model embeddings for semantic deduplication. It is another whole research question with scaling, cost, ablation experiments, and trade-off with near-deduplication. There are some intriguing takes on this[[20]](#20), but we still need more situated evidence to draw a conclusion (e.g, [@abbas_2023](http://arxiv.org/abs/2303.09540)'s only text deduplication reference is [@lee_2022a](http://arxiv.org/abs/2107.06499), whose main claim is deduplicating helps instead of trying to be SOTA).
4. Optimization. There is always room for optimization: better quality evaluation, scaling, downstream performance impact analysis etc.
5. Then there is another direction to look at things: To what extent near-deduplication starts to hurt performance? To what extent similarity is needed for diversity instead of being considered as redundancy?

## Credits

The banner image contains emojis (hugging face, Santa, document, wizard, and wand) from Noto Emoji (Apache 2.0). This blog post is proudly written without any generative APIs.

Huge thanks to Huu Nguyen @Huu and Hugo Laurençon @HugoLaurencon for the collaboration in BigScience and everyone at BigCode for the help along the way! If you ever find any error, feel free to contact me: mouchenghao at gmail dot com.

## Supporting Resources

- [Datasketch](https://github.com/ekzhu/datasketch) (MIT)
- [simhash-py](https://github.com/seomoz/simhash-py/tree/master/simhash) and [simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
- [Deduplicating Training Data Makes Language Models Better](https://github.com/google-research/deduplicate-text-datasets) (Apache 2.0)
- [Gaoya](https://github.com/serega/gaoya) (MIT)
- [BigScience](https://github.com/bigscience-workshop) (Apache 2.0)
- [BigCode](https://github.com/bigcode-project) (Apache 2.0)

## References

- <a id="1">[1]</a> : Nikhil Kandpal, Eric Wallace, Colin Raffel, [Deduplicating Training Data Mitigates Privacy Risks in Language Models](http://arxiv.org/abs/2202.06539), 2022
- <a id="2">[2]</a> : Gowthami Somepalli, et al., [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](http://arxiv.org/abs/2212.03860), 2022
- <a id="3">[3]</a> : Katherine Lee, Daphne Ippolito, et al., [Deduplicating Training Data Makes Language Models Better](http://arxiv.org/abs/2107.06499), 2022
- <a id="4">[4]</a> : Loubna Ben Allal, Raymond Li, et al., [SantaCoder: Don't reach for the stars!](http://arxiv.org/abs/2301.03988), 2023
- <a id="5">[5]</a> : Leo Gao, Stella Biderman, et al., [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](http://arxiv.org/abs/2101.00027), 2020
- <a id="6">[6]</a> : Asier Gutiérrez-Fandiño, Jordi Armengol-Estapé, et al., [MarIA: Spanish Language Models](http://arxiv.org/abs/2107.07253), 2022
- <a id="7">[7]</a> : Jack W. Rae, Sebastian Borgeaud, et al., [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](http://arxiv.org/abs/2112.11446), 2021
- <a id="8">[8]</a> : Xi Victoria Lin, Todor Mihaylov, et al., [Few-shot Learning with Multilingual Language Models](http://arxiv.org/abs/2112.10668), 2021
- <a id="9">[9]</a> : Hugo Laurençon, Lucile Saulnier, et al., [The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset](https://openreview.net/forum?id=UoEw6KigkUn), 2022
- <a id="10">[10]</a> : Daniel Fried, Armen Aghajanyan, et al., [InCoder: A Generative Model for Code Infilling and Synthesis](http://arxiv.org/abs/2204.05999), 2022
- <a id="11">[11]</a> : Erik Nijkamp, Bo Pang, et al., [CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis](http://arxiv.org/abs/2203.13474), 2023
- <a id="12">[12]</a> : Yujia Li, David Choi, et al., [Competition-Level Code Generation with AlphaCode](http://arxiv.org/abs/2203.07814), 2022
- <a id="13">[13]</a> : Frank F. Xu, Uri Alon, et al., [A Systematic Evaluation of Large Language Models of Code](http://arxiv.org/abs/2202.13169), 2022
- <a id="14">[14]</a> : Aakanksha Chowdhery, Sharan Narang, et al., [PaLM: Scaling Language Modeling with Pathways](http://arxiv.org/abs/2204.02311), 2022
- <a id="15">[15]</a> : Lewis Tunstall, Leandro von Werra, Thomas Wolf, [Natural Language Processing with Transformers, Revised Edition](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/), 2022
- <a id="16">[16]</a> : Denis Kocetkov, Raymond Li, et al., [The Stack: 3 TB of permissively licensed source code](http://arxiv.org/abs/2211.15533), 2022
- <a id="17">[17]</a> : [Rocky | Project Hail Mary Wiki | Fandom](https://projecthailmary.fandom.com/wiki/Rocky)
- <a id="18">[18]</a> : Raimondas Kiveris, Silvio Lattanzi, et al., [Connected Components in MapReduce and Beyond](https://doi.org/10.1145/2670979.2670997), 2014
- <a id="19">[19]</a> : Jacob Austin, Augustus Odena, et al., [Program Synthesis with Large Language Models](http://arxiv.org/abs/2108.07732), 2021
- <a id="20">[20]</a>: Amro Abbas, Kushal Tirumala, et al., [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](http://arxiv.org/abs/2303.09540), 2023
- <a id="21">[21]</a>: Edith Cohen, [MinHash Sketches : A Brief Survey](http://www.cohenwang.com/edith/Surveys/minhash.pdf), 2016
