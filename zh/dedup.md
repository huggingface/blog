---
title: "BigCode 背后的大规模数据去重"
thumbnail: /blog/assets/dedup/thumbnail.png
authors:
  - user: chenghao
translators:
  - user: MatrixYao
  - user: zhongdongy
    proofreader: true
---

# BigCode 背后的大规模数据去重


## 目标受众

本文面向对大规模文档去重感兴趣，且对散列 (hashing) 、图 (graph) 及文本处理有一定了解的读者。

## 动机

老话说得好: 垃圾进，垃圾出 (garbage in, garbage out)，把数据处理干净再输入给模型至关重要，至少对大语言模型如此。虽然现在一些明星大模型 (严格来讲，它们很多是 API) 的存在让大家恍惚产生了数据质量好像不那么重要了的错觉，但事实绝非如此。

在 BigScience 和 BigCode 项目中，在数据质量方面，我们面临的一个很大的问题是数据重复，这不仅包括训练集内的数据重复，还包括训练集中包含测试基准中的数据从而造成了基准污染 (benchmark contamination)。已经有研究表明，当训练集中存在较多重复数据时，模型倾向于逐字输出训练数据 [[1]](#1) (这一现象在其他一些领域并不常见 [[2]](#2))，而且训得的模型也更容易遭受隐私攻击 [[1]](#1)。除了能避免上面两个问题外，去重还有不少好处:

1. 让训练更高效: 你可以用更少的训练步骤获得相同的，甚至是更好的性能 [[3]](#3) [[4]](#4)。
2. 防止可能的数据泄漏和基准污染: 数据重复会损害你的模型性能报告的公信力，并可能让所谓的改进沦为泡影。
3. 提高数据可得性。我们大多数人都负担不起重复下载或传输数千 GB 文本的成本，更不用说由此带来的额外训练成本了。对数据集进行去重，能使其更易于学习、传输及协作。

## 从 BigScience 到 BigCode

我想先分享一个故事，故事主要讲述我如何接受数据去重这一任务，过程如何，以及在此过程中我学到了什么。

一切开始于 LinkedIn 上的一次对话，当时 [BigScience](https://bigscience.huggingface.co/) 已经开始几个月了。Huu Nguyen 注意到我在 GitHub 上的一个小项目并找到了我，问我是否有兴趣为 BigScience 做数据去重工作。我当然愿意了，尽管当时我完全没意识到由于数据量巨大，这项工作比想象中麻烦很多。

这项工作既有趣又充满挑战。挑战在于，我对处理如此大规模的数据并没有太多经验。但项目组的每个人仍然欢迎我、信任我，还给了我数千美元的云计算预算。有多少回，我不得不从睡梦中醒来，反复确认我是否关闭了那些云实例。我不停地在试验和错误中学习，在此过程中，新的视角被打开了。如果没有 BigScience，可能我永远不会有这种视角。

一年后的今天，我正在把从 BigScience 学到的东西应用到 [BigCode](https://www.bigcode-project.org/) 项目中去，去处理更大的数据集。除了英语 [[3]](#3) LLM 之外，我们已经再次证明数据去重也能改进代码模型 [[4]](#4) 的性能。有了数据去重，我们可以用更小的数据集达到更优的性能。现在，亲爱的读者，我想与你分享我学到的知识，希望你能透过数据去重的镜头一瞥 BigCode 项目的幕后故事。

下表列出了 BigScience 项目中各数据集使用的去重方法，以供参考:

| 数据集                              | 输入数据量                       | 输出数据尺寸或数据精简比                                        | 去重粒度                 | 方法                                        | 参数                                                       | 语种     | 耗时                |
| ------------------------------------ | -------------------------------- | --------------------------------------------------------------- | --------------------- | --------------------------------------------- | ---------------------------------------------------------------- | ------------ | ------------------- |
| OpenWebText2[[5]](#5)                | 对 URL 去重后: 193.89 GB（69M）| 使用 MinHash LSH 后: 65.86 GB（17M）                               | URL + 文档        | URL（精确匹配）+ 文档（MinHash LSH）           | $(10, 0.5, ?, ?, ?)$                                       | 英语      |                     |
| Pile-CC[[5]](#5)                     | *~306 GB*                        | *227.12 GiB（~55M）*                                             | 文档              | 文档（MinHash LSH）                         | $(10, 0.5, ?, ?, ?) $                                      | 英语      | 数天      |
| BNE5[[6]](#6)                        | 2 TB                              | 570 GB                                                          | 文档              | Onion                                         | 5-元组                                                           | 西班牙语      |                     |
| MassiveText[[7]](#7)                 |                                  | 0.001 TB ~ 2.1 TB                                               | 文档              | 文档（精确匹配 + MinHash LSH）                | $(?, 0.8, 13, ?, ?)$                                       | 英语      |                     |
| CC100-XL[[8]](#8)                    |                                  | 0.01 GiB ~ 3324.45 GiB                                          | URL + 段落       | URL（精确匹配） + 段落（精确匹配）                 | SHA-1                                                            | 多语种 |                     |
| C4[[3]](#3)                          | 806.92 GB (364M)                 | 3.04% ~ 7.18% **↓** （训练集）                                     | 子字符串或文档 | 子字符串（后缀数组）或文档（MinHash）  | 后缀数组：50-词元，MinHash: $(9000, 0.8, 5, 20, 450)$ | 英语      |                     |
| Real News[[3]](#3)                   | ~120 GiB                         | 13.63% ~ 19.4% **↓**（训练集）                                  | 同 **C4**        | 同 **C4**                                | 同 **C4**                                                   | 英语      |                     |
| LM1B[[3]](#3)                        | ~4.40 GiB（30M）                  | 0.76% ~ 4.86% **↓**（训练集）                                   | 同 **C4**        | 同 **C4**                                | 同 **C4**                                                   | 英语      |                     |
| WIKI40B[[3]](#3)                     | ~2.9M                            | 0.39% ~ 2.76% **↓**（训练集）                                    | 同 **C4**        | 同 **C4**                                | 同 **C4**                                                   | 英语      |                     |
| BigScience ROOTS 语料集[[9]](#9) |                                  | 0.07% ~ 2.7% **↓** (文档) + 10.61% ~ 32.30% **↓** (子字符串) | 文档 + 子字符串  | 文档 (SimHash) + 子字符串 (后缀数组) | SimHash：6-元组，汉明距离（hamming distance）为 4，后缀数组：50-词元  | 多语种 | 12 小时 ~ 数天 |

下表是我们在创建 BigCode 的训练数据集 (训练数据皆为代码) 时所用的方法。这里，如果当遇到没有名字的数据集时，我们就用模型名称来代替。

| 模型                 | 去重方法               | 参数                             | 去重级别    |
| --------------------- | -------------------- | -------------------------------------- | -------- |
| InCoder[[10]](#10)    | 精确匹配                | 代码词元/MD5 + 布隆滤波（Bloom filtering） | 文档 |
| CodeGen[[11]](#11)    | 精确匹配                | SHA256                                 | 文档 |
| AlphaCode[[12]](#12)  | 精确匹配                | 忽略空格                      | 文档 |
| PolyCode[[13]](#13)   | 精确匹配                | SHA256                                 | 文档 |
| PaLM Coder[[14]](#14) | Levenshtein 距离 |                                        | 文档 |
| CodeParrot[[15]](#15) | MinHash + LSH        | $(256, 0.8, 1)$                  | 文档 |
| The Stack[[16]](#16)  | MinHash + LSH        | $(256, 0.7, 5)$                  | 文档 |

MinHash + LSH 参数 $(P, T, K, B, R)$ :

1. $P$ 哈希函数的个数或排列的个数
2. $T$ Jaccard 相似度阈值
3. $K$ K- 元组
4. $B$ 条带数
5. $R$ 每条带包含的行数

我们做了一个简单的演示程序来说明这些参数对结果的影响: [MinHash 数学演示](https://huggingface.co/spaces/bigcode/near-deduplication)。

## 例解 MinHash

在本节中，我们将详细介绍在 BigCode 中使用的 MinHash 方法的每个步骤，并讨论该方法的系统扩展性问题及其解决方案。我们以一个含有三个英文文档为例来演示整个工作流程:

| doc_id | 内容                                  |
| ------ | ---------------------------------------- |
| 0      | Deduplication is so much fun!            |
| 1      | Deduplication is so much fun and easy!   |
| 2      | I wish spider dog[[17]](#17) is a thing. |

MinHash 的典型工作流程如下:

1. 词袋生成 (生成 n- 元组) 及指纹生成 (生成 MinHash): 将每个文档映射成一组哈希值。
2. 局部敏感哈希 (LSH): 逐条带 (band) 的比较文档的相似性，并将相似的文档聚类以减少后续比较的次数。
3. 去重: 决定保留或删除哪些重复文档。

### 词袋生成

与大多数文本应用一样，我们需要先把文本表示成词袋，这里我们通常使用 N- 元组词袋。在本例中，我们使用以单词为基本单元的 3- 元组 (即每 3 个连续单词组成一个元组)，且不考虑标点符号。我们后面会回过头来讨论元组大小对性能的影响。

| doc_id | 3-元组   |
| ------ | ------------------------------------------------------------------------------- |
| 0      | {"Deduplication is so", "is so much", "so much fun"}                            |
| 1      | {'so much fun', 'fun and easy', 'Deduplication is so', 'is so much'}            |
| 2      | {'dog is a', 'is a thing', 'wish spider dog', 'spider dog is', 'I wish spider'} |

这个操作的时间复杂度为 $\mathcal{O}(NM)$，其中 $N$ 表示文档数，而 $M$ 表示文档长度。也就是说，时间复杂度与数据集大小呈线性关系。我们可以用多进程或分布式计算来并行化词袋生成过程。

### 指纹计算

使用 MinHash 方法时，每个 N- 元组需要生成多个哈希值，此时我们通常要么 1) 使用不同的哈希函数进行多次哈希，要么 2) 使用一个哈希函数进行哈希后再进行多次重排。本例中，我们选择第二种方法，重排生成 5 个哈希值。 更多 MinHash 的变体可以参考 [MinHash - 维基百科](https://en.wikipedia.org/wiki/MinHash?useskin=vector)。

| N-元组             | 哈希值                                             |
| ------------------- | ----------------------------------------------------------- |
| Deduplication is so | [403996643, 2764117407, 3550129378, 3548765886, 2353686061] |
| is so much          | [3594692244, 3595617149, 1564558780, 2888962350, 432993166] |
| so much fun         | [1556191985, 840529008, 1008110251, 3095214118, 3194813501] |

对以上文档哈希矩阵中的每一列取最小值 —— 即  “MinHash” 中的 “Min” 的题中之义，我们就能得到该文档最终的 MinHash 值:

| doc_id | MinHash                                                    |
| ------ | ---------------------------------------------------------- |
| 0      | [403996643, 840529008, 1008110251, 2888962350, 432993166]  |
| 1      | [403996643, 840529008, 1008110251, 1998729813, 432993166]  |
| 2      | [166417565, 213933364, 1129612544, 1419614622, 1370935710] |

从技术上讲，虽然我们通常取最小值，但这并不代表我们一定要取每列的最小值。其他顺序统计量也是可以的，例如最大值、第 k 个最小值或第 k 个最大值 [[21]](#21)。

在具体实现时，我们可以使用 `numpy` 来对这些操作进行向量化。该操作的时间复杂度为 $\mathcal{O}(NMK)$，其中 $K$ 是排列数。以下列出了我们的代码，它是基于 [Datasketch](https://github.com/ekzhu/datasketch) 的实现修改而得的。

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

熟悉 [Datasketch](https://github.com/ekzhu/datasketch) 的读者可能会问，为什么我们要费心费力剥离 Datasketch 库提供的所有高级功能？其主要原因并不是因为我们要减少依赖项，而是因为我们想要尽可能地榨取 CPU 的算力。而将多个步骤融合到一个函数中，是更好利用计算资源的手段之一。

由于每个文档的计算互相独立，因此我们可以充分利用 `datasets` 库的 `map` 函数来实现并行化:

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

指纹计算完毕之后，每个文档都被映射成了一个整数数组。为了弄清楚哪些文档彼此相似，我们需要根据这些指纹对它们进行聚类。轮到 **局部敏感哈希 (Locality Sensitive Hashing，LSH)** 闪亮登场了。

### 局部敏感哈希 (LSH)

LSH 将指纹数组按行分成若干个条带 (band)，每个条带的行数相同，如果遇到最后一个条带行数不足，我们就直接忽略它。以条带数 $b=2$ 为例，每个条带有 $r=2$ 行，具体组织如下:

| doc_id | MinHash                                                    | 条带                                                  |
| ------ | ---------------------------------------------------------- | ------------------------------------------------------ |
| 0      | [403996643, 840529008, 1008110251, 2888962350, 432993166]  | [0:[403996643, 840529008], 1:[1008110251, 2888962350]] |
| 1      | [403996643, 840529008, 1008110251, 1998729813, 432993166]  | [0:[403996643, 840529008], 1:[1008110251, 1998729813]] |
| 2      | [166417565, 213933364, 1129612544, 1419614622, 1370935710] | [0:[166417565, 213933364], 1:[1129612544, 1419614622]] |

若两个文档在某条带上 MinHash 值相同，这两个文档就会被聚到同一个桶中备选。

| 条带 ID     | 条带值                   | doc_ids |
| ---------- | ------------------------ | ------- |
| 0          | [403996643, 840529008]   | 0, 1    |
| 1          | [1008110251, 2888962350] | 0       |
| 1          | [1008110251, 1998729813] | 1       |
| 0          | [166417565, 213933364]   | 2       |
| 1          | [1129612544, 1419614622] | 2       |

遍历 `doc_ids` 列的每一行，将其中的文档两两配对就生成了候选对。上表中，我们能生成一个候选对: `(0, 1)` 。

### 候选对生成后 ……

很多数据去重的论文或教程讲完上一节就结束了，但在实际项目中我们还涉及如何处理这些候选对的问题。通常，候选对生成后，我们有两个选择:

1. 由于 MinHash 只是一个近似，所以仍需计算两个文档的 N- 元组集合的交并比来算得准确的 Jaccard 相似性。此时，因为 LSH 已经帮我们过滤了不少，所以最终参与计算的候选对的量会大大减少。在 BigCode 项目中，我们起初就采用了这种做法，效果相当不错。
2. 我们还可以直接认可 LSH 选出来的相似对。这里面可能会有个问题: Jaccard 相似性不具传递性，也就是说 $A$ 相似于 $B$ 且  $B$ 相似于 $C$，并不意味着 $A$ 相似于 $C$。所以这里可能会有不少假阳性。通过在 The Stack 数据集上的实验，我们发现，直接认可 LSH 选出来的相似对在很大程度上能提高下游模型的性能，同时还节省了处理时间和训练时间。因此目前我们正慢慢开始转向这种方法。但是，这个经验并不是放之四海而皆准的，如果你准备在自己的数据集上仿效我们的做法，我们建议你在此之前好好检查你的数据集及其特点，然后作出数据驱动的决策。

最后，我们可以用生成的相似文本对构建一个图，在这个图中，重复的文档会被聚至同一个社区或同一个连通子图中。不幸的是， `datasets` 在这方面帮不上什么忙，因为现在我们需要类似 `groupby` 的功能，以根据 _条带 ID_ 及 _文档在该条带上的取值_ 对文档进行聚类。下面列出了我们尝试过的一些方案:

**方案 1: 老办法，迭代数据集以创建图，然后用一个图处理库对其做社区检测或者连通分量检测。**

我们测试下来，该方案的扩展性不怎么好，其原因是多方面的: 首先，整个数据集迭代起来很慢，而且内存消耗很大; 其次，诸如 `graphtool` 或 `networkx` 的市面上流行的图处理库创建图的开销较大。

**方案 2: 使用流行的 Python 框架 (如 `dask` ) 及其高效的 `groupby` 操作**。

但迭代慢和创建图慢的问题仍然存在。

**方案 3: 迭代数据集并使用并查集 (union find data structure) 对文档进行聚类。**

这个方案引入了一个很小的迭代开销，对中等数据集的有不错的效果不错，但在大数据集上还是慢。

```python
for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
	for cluster in table.values():
		if len(cluster) <= 1:
			continue
		idx = min(cluster)
		for x in cluster:
			uf.union(x, idx)
```

**方案 4: 对大数据集，使用 Spark。**

我们已经知道到 LSH 的有些步骤是可以并行化的，我们可以用 Spark 来实现它们。Spark 的好处是，它开箱即支持分布式 `groupBy` ，而且也能很轻松地实现像 [[18]](#18) 这样的连通分量检测算法。注意，这里我们并没有使用 Spark 的原生 MinHash 实现，其原因是迄今为止我们所有的实验都源于 [Datasketch](https://github.com/ekzhu/datasketch)，而 Datasketch 的 MinHash 实现与 Spark 的原生实现完全不同。我们希望之前的经验和教训能帮助到后面的工作，而不是另起炉灶，进入另一个消融实验的轮回，因此我们选择在 Spark 中自己实现 Datasketch 的 MinHash 算法。

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
	.groupBy(lambda x:(x[0], x[1]))
	.flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
	.distinct()
	.cache()
)
```

以下是基于 [[18]](#18) 的简单连通分量检测算法的 Spark 实现。

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

多亏了云计算提供商，我们可以使用 GCP DataProc 等服务轻松地搭建 一个 Spark 集群。 **最终，我们把程序运行起来，只用了不到 4 小时就完成了 1.4 TB 数据的去重工作，每小时仅需 15 美元。**

## 数据质量很重要

我们不可能爬着梯子登上月球。因此我们不仅要确保方向正确，还要确保方法正确。

早期，我们使用的参数主要来自 CodeParrot 的实验，消融实验表明这些参数确实提高了模型的下游性能 [[16]](#16)。后来，我们开始沿着这条路进一步探索，由此进一步确认了以下结论 [[4]](#4):

1. 数据去重可以在缩小数据集 (6 TB VS. 3 TB) 规模的同时提高模型的下游性能
2. 虽然我们还没有完全搞清楚其能力边界及限制条件，但我们确实发现更激进的数据去重 (6 TB VS. 2.4 TB) 可以进一步提高性能，方法有:
  1. 降低相似度阈值
  2. 使用更长的元组 (如: 一元组 → 五元组)
  3. 放弃误报检查，承受一小部分误报带来的数据损失

![1- 元组时不同设置影响的小提琴图](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/violin_chart_1.png)

![5- 元组时不同设置影响的小提琴图](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/violin_chart_2.png)

<center>
图例: 上述两幅图展示了相似性阈值和元组大小带来的影响，第一幅图使用 1- 元组，第二幅图使用 5- 元组。红色虚线表示相似性阈值: 低于该值的文档与同一簇中其他文档的相似性低于阈值，我们将其视为误报。
</center>

上面两幅图可以帮助我们理解为什么有必要仔细检查 CodeParrot 以及早期版本的 The Stack 训练数据上的误报: 这是使用 1- 元组的误报比例会很大; 上图还表明，将元组大小增加到 5，误报比例会显著降低。如果想激进点去重的话，阈值可以设低点。

还有实验表明，降低阈值会删除更多包含部分相似内容的文档，因此意味着提高了我们最想删除的那部分文档的查全率。

## 系统扩展性

![Scaling results for dataset size and deduplication time](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/scale.png)

<center> 图例: 数据去重时间与原始数据集规模的关系。测试基于 GCP 上的 15 个 c2d-standard-16 实例，每个实例每小时的成本约为 0.7 美元。</center>

![CPU usage screenshot for the cluster during processing JSON dataset](https://huggingface.co/datasets/chenghao/dedup_blog_assets/resolve/main/data/usage.png)

<center> 图例: 集群在处理 JSON 数据集时的 CPU 使用率。</center>

上述扩展性数据未必非常严格，但也足够说明，在给定预算的情况下，数据去重耗时与数据集规模的关系应该是线性的。如果你仔细看一下处理 JSON 数据集 (The Stack 数据集的最大子集) 的集群资源使用情况，你会发现实际总计算时间 (图中第 2 和第 3 阶段) 主要都花在了 MinHash + LSH (图中第 2 阶段) 上，这与我们先前的分析一致，即第 2 阶段 d 的时间复杂度为 $ \mathcal{O}(NM) $ — 与数据体量成线性关系。

## 谨慎行事

数据去完重并不意味着万事大吉了，你仍然需要对数据进行彻底的探索和分析。此外，上文这些有关数据去重的发现来自于 The Stack 数据集，并不意味着它能无脑适用于其他数据集或语言。要构建一个好的训练数据集，我们仅仅迈出了万里长征的第一步，后面还有很多工作要做，例如数据质量过滤 (如过滤漏洞数据、毒性数据、偏见数据、模板生成的数据、个人身份数据等)。

我们还鼓励你在训练前像我们一样对数据集进行彻底的分析，因为大家的情况可能各不相同。例如，如果你的时间和计算预算都很紧张，那么数据去重可能不是很有帮助: [@geiping_2022](http://arxiv.org/abs/2212.14034) 提到基于子字符串的数据去重并没有提高他们模型的下游性能。在使用前，可能还需要对现存数据集进行彻底检查，例如，[@gao_2020](http://arxiv.org/abs/2101.00027) 声明他们只确保 Pile 本身及其子集都已去重，但不保证其与任何下游基准数据集没有重复，要不要对 Pile 与下游基准数据集进行去重取决于使用者自己。

在数据泄露和基准污染方面，还有很多需要探索的地方。由于 HumanEval 也是 GitHub Python 存储库之一，我们不得不重新训练了我们的代码模型。早期的工作还发现，最流行的编码基准之一的 MBPP[[19]](#19) 与许多 Leetcode 问题有很多相似之处 (例如，MBPP 中的任务 601 基本上是 Leetcode 646，任务 604 ≃ Leetcode 151)。我们都知道 GitHub 中不乏很多编程挑战赛题及其答案代码。如果居心叵测的人把所有基准测试的 Python 代码以不易察觉的方式上传到 Github，污染你所有的训练数据，这事儿就更难了。

## 后续方向

1. 子串去重。尽管在英语 [[3]](#3) 上子串去重是有益的，但尚不清楚是否对代码数据也有用;
2. 重复段落: 在一篇文档中重复多次的段落。 [@rae_2021](http://arxiv.org/abs/2112.11446) 分享了一些关于如何检测和删除它们的有趣的启发式方法。
3. 使用模型嵌入进行语义级的去重。这是另外一套思路了，需要一整套去重、扩展性、成本、销蚀等各方面的实验和权衡。对此 [[20]](#20) 提出了一些有趣的看法，但我们仍然需要更多实际证据才能得出结论 (其文本去重工作仅参考了 [@lee_2022a](http://arxiv.org/abs/2107.06499) 的工作，而 [@lee_2022a](http://arxiv.org/abs/2107.06499) 的主张主要是去重有作用而并未证明其效果达到了 SOTA)。
4. 优化。还有不少优化空间: 更好的质量评估标准、扩展性、对下游性能影响的分析等。
5. 换个角度: 对相似数据，去重到什么程度就会开始损害性能？需要保留多少相似数据以保留数据的多样性又不至冗余？

## 致谢

题图中的表情符 (Hugging Face、圣诞老人、文档、巫师以及魔杖) 来自于 Noto Emoji (Apache 2.0)。我也庄严保证，这篇博文是我一个字一个字敲出来的，没有使用任何文本生成 API。

非常感谢 Huu Nguyen(@Huu) 和 Hugo Laurençon(@HugoLaurencon) 在 BigScience 项目中的合作，以及 BigCode 项目中每个人一路上的帮助！如果你发现任何错误，请随时联系我: mouchenghao at gmail dot com。

## 更多资源

- [Datasketch](https://github.com/ekzhu/datasketch) (MIT)
- [simhash-py](https://github.com/seomoz/simhash-py/tree/master/simhash) 及 [simhash-cpp](https://github.com/seomoz/simhash-cpp) (MIT)
- [Deduplicating Training Data Makes Language Models Better](https://github.com/google-research/deduplicate-text-datasets) (Apache 2.0)
- [Gaoya](https://github.com/serega/gaoya) (MIT)
- [BigScience](https://github.com/bigscience-workshop) (Apache 2.0)
- [BigCode](https://github.com/bigcode-project) (Apache 2.0)

## 参考文献

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
