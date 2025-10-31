---
title: "Data Analytics with Apache Arrow and Hugging Face Datasets" 
thumbnail: /blog/assets/arrow-datasets/bla.png
authors:
- user: cakiki
- user: lhoestq
- user: mariosasko
---

# Data Analytics with Apache Arrow and Hugging Face Datasets

<!-- {blog_metadata} -->
<!-- {authors} -->

Hugging Face Datasets is a library that was designed to power machine learning training workflows. It enables users to share, download, process, and format datasets to get them ready for training with any of the major machine learning frameworks, across all data modalities: audio, image, text, and tabular. In the Hugging Face ecosystem, the 🤗 `datasets` library is typically the first entry point of a machine learning project: it is therefore especially crucial it be performant and efficient.

| ![HF Libraries](./assets/arrow-datasets/hf-libraries.png) |
|:--:|
| <i>Hugging Face libraries mapped to the steps of a typical machine learning workflow. Source: <a href="https://github.com/nlp-with-transformers" rel="noopener" target="_blank" >Natural Language Processing with Transformers</a></i>|

One of the many strengths of 🤗 `datasets` is its ability to [quickly and efficiently](https://huggingface.co/docs/datasets/about_arrow) handle data that is too large fit into system memory. This is made possible by using the Apache [Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html) as its underlying memory model. This post will showcase the synergies between the two projects, and how using the Arrow ecosystem can supercharge your library, giving you access to more features than you initially bargained for. This post will show you how you can use 🤗 `datasets` for out-of-core data analytics to better understand your data, ahead of using it to train a model.

## Apache Arrow

According to the [official website](https://arrow.apache.org/), Apache Arrow is: (1) a column-oriented **standardized memory format**, (2) a set of **libraries** that implement said format and various utilities around it, and (3) an **ecosystem** of projects using them. 🤗 `datasets` is one of many projects of this ecosystem and leverages the Arrow memory format through the use of the [PyArrow library](https://arrow.apache.org/docs/python/index.html): the official Python API of Apache Arrow. The Arrow format is further described as a:

> language-independent columnar memory format for flat and hierarchical data, organized for efficient analytic operations on modern hardware like CPUs and GPUs. The Arrow memory format also supports zero-copy reads for lightning-fast data access without serialization overhead.

Let's unpack this:

- **language independent**: the data representation in memory is the same, no matter the programming language or library.
- **columnar**: data is contiguous in memory and grouped by column, as opposed to by row.
- **efficient analytic operations**: the data is formatted in such a way that allows vectorized operations like [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
- **zero-copy reads with serialization overhead**: once data is in memory, moving it between different libraries (e.g. from `pandas` to `polars`), threads, or even different languages (e.g. from `python` to `rust`) is free.

Further features of Arrow that make it especially useful in machine learning worflows include:

- **Constant-time random access**: indexing into a specific row is a constant time operation
- **Efficient sequential scans**: the columnar memory layout makes for efficient iterating over the dataset
- **Data can be memory-mapped**: larger-than-memory datasets can be serialized onto disk and memory-mapped from there.

As a user, you typically won't need interact with Arrow directly, but rather use [libaries which use Arrow](https://arrow.apache.org/powered_by/)—such as Hugging Face Datasets—to represent tabular data in memory. However, as we'll see later in this post, even if the Arrow internals are abstracted away, being aware of them can give users access to more functionality than afforded by a given library.

| ![Copy](./assets/arrow-datasets/copy.png) | ![Standardization](./assets/arrow-datasets/shared.png)
|:--:|:--:|
|<i>In the absence of a standard, considerable overhead goes into moving, copying, and serializing data between languages and frameworks</i>|<i>Loading data into memory is standardized by the Arrow format</i>|
<center><i>Inspired by the visuals of the <a href="https://arrow.apache.org/overview/" rel="noopener" target="_blank" >Apache Arrow Overview</a></i></center>

### Hugging Face Datasets

A Hugging Face `datasets.Dataset` object is at its core a dataframe-like structure called an Arrow Table: a 2-dimensional data structure with a schema and named columns. When you load a dataset for the first time, it is serialized in batches into one ore more `.arrow` memory files in your cache directory, then finally mmapped from those files on disk. To peek behind the scenes of this process let's begin by downloading the dataset we'll be using, a subset of [the Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup), a dataset that was used to train the [StarCoder](https://huggingface.co/bigcode/starcoder) family of code large language models. Each row of this dataset corresponds to one text file from GitHub, along with the corresponding metadata—programming language, license, number of stars, and file size to name a few.

```python
>>> from datasets import load_dataset
>>> dset = load_dataset("cakiki/stack-smol-xxl", split="train") # ~107GB of disk space required to run this
>>> type(dset)
<class 'datasets.arrow_dataset.Dataset'> 
>>> type(dset.data.table) # dset.data.table is the underlying Arrow Table
<class 'pyarrow.lib.Table'>
>>> [shard["filename"].split("/")[-1] for shard in dset.cache_files]
['parquet-train-00000-of-00144.arrow', 'parquet-train-00001-of-00144.arrow', ... ] # These are the serialized files on disk. (Output trimmed) 
>>> print(len(dset.cache_files), "cache files")
144 cache files
```
The data is saved in the same format it exists in memory which allows fast access with zero-copy. However the data columns aren't compressed which makes the format less suited for long term disk storage than its sister project: [**Apache Parquet**](https://parquet.apache.org/). Parquet is also a columnar data format, that focuses on efficient disk storage and retrieval, and it achieves this through efficiently compression each data column. Serializing from Parquet to Arrow and vice-versa is very easy, and the two projects are generally very closely related, so much so for example, that the C++ implementation of Parquet is now part of the Arrow project; but enough of this short interlude, and let's go back to our cached dataset.

Caching datasets onto disk is not limited to loading data, but also transforming it. Indeed, as we'll see later in this post, when you operate on a 🤗 dataset—be it through filtering, mapping, or structural transformations—these operations are also cached to disk and only need to run once. Before we move on to the next section, let's quickly inspect the code dataset we just loaded:

```python
>>> print(f'{dset.num_rows:,}', "rows")
11,658,586 rows
>>> print(dset.num_columns, "columns")
29 columns
>>> print(f'{dset.dataset_size / 1024**3:.2f}', " GB")
73.18 GB
>>> dset[1200000]
'hexsha': '7a4d22966630185a027c6ee50494472b3c1b3fa4', 'size': 10253, 'ext': 'jl', 'lang': 'Julia', 'max_stars_repo_path': 'src/cwv.jl' ...} # Indexing into a dataset returns the correspdonding row as a dictionary.
>>> dset[:4]
{'hexsha': ['0a84ade1a79baf9000df6f66818fc3a568c24173', '0a84b588e484c6829ab1a6706db302acd8514288', '0a89bca33bdfce1f07390d394915f18e0d901651', '0a96a3eeaa089f8582180f803252e3739ab2661d'], 'size': [10354, 999, 43147, 597], 'ext': ['css', 'css', 'css', 'css'], 'lang': ['CSS', 'CSS', 'CSS', 'CSS'], ...} # By default, slicing a dataset returns a dictionary of the columns as lists of standard Python objects. Makes sense since arrow is a columnar format. (Output trimmed)
```

### Hugging Face Dataset Formats

As you just saw, accessing data by slicing returns a dictionary where each column is a list of Python objects, and accessing a single row returns a dictionary of the corresponding row. This dictionary format is the default way data is returned when accessing data  in the datasets library. This allows users to write pure Python `.map()` and `.filter()` operations that operate on Python objects. 

We are however not limited to Python objects, and can choose to have our `datasets.Dataset` object to return any of the formats supported by the `PyArrow` bindings, and those include `pandas` dataframes, `NumPy` arrays, tensors, and even Arrow Tables.

```python
>>> dset.set_format("numpy")
>>> print(dset[:4])

>>> dset.set_format("pandas")
>>> print(dset[:4])

>>> dset.set_format("arrow")
>>> print(dset[:4])
pyarrow.Table
hexsha: string
size: int64
...
----
hexsha: [["0a84ade1a79baf9000df6f66818fc3a568c24173","0a84b588e484c6829ab1a6706db302acd8514288","0a89bca33bdfce1f07390d394915f18e0d901651","0a96a3eeaa089f8582180f803252e3739ab2661d"]]
size: [[10354,999,43147,597]]
ext: [["css","css","css","css"]]
lang: [["CSS","CSS","CSS","CSS"]]
...
>>> dset.set_format(None)
```
Changing the format of a dataset also changes the behavior of the `.map()` operation in batched mode. Since each batch is effectively a slice of the dataset, you can now define your mapping functions to operate on any of the formats we just mentioned. This means you can leverage functions that operate on `NumPy` arrays, `pandas` Dataframes, and Arrow Tables.

## Mapping Compute Primitives
Beyond providing an efficient memory format and ways to serialize your data into this format, Apache Arrow also provides ways to peform computation on that data through a series of highly performant C++ primitives unified under the **Arrow Compute API**. These include aggregations, grouped aggregations, scalar functions, vector functions, and selections—to name a few. A full list of available functions can be found [here](https://arrow.apache.org/docs/python/compute.html).

More practically, this means we can use any of these functions with our `.map()` operations, so long as we make sure we're operating on Arrow Tables, so let's start by making sure of that:
```python
>>> dset.set_format("arrow")
```
Let's now import the Compute API:
```python
import pyarrow.compute as pc
```
We can start by seeing what programming languages are included in our dataset:
```python
>>> %time pc.unique(dset["lang"]).to_pylist()
['CSS', 'Prolog', 'C', 'FORTRAN', 'Solidity', 'Kotlin', 'Literate Agda', 'Julia', 'Java Server Pages', 'Isabelle', 'Idris', 'Lean', 'PowerShell', 'Go', 'Erlang', 'F#', 'Ada', 'Pascal', 'Perl', 'R', 'Protocol Buffer', 'CMake', 'SAS', 'Ruby', 'Rust', 'RMarkdown', 'C#', 'Smalltalk', 'Haskell', 'Maple', 'Mathematica', 'OCaml', 'Makefile', 'Lua', 'Literate CoffeeScript', 'Literate Haskell', 'reStructuredText', 'Racket', 'Standard ML', 'SystemVerilog', 'TeX', 'Awk', 'Assembly', 'Alloy', 'Agda', 'Emacs Lisp', 'Dart', 'Cuda', 'Bluespec', 'Augeas', 'Batchfile', 'Tcsh', 'Stan', 'Scala', 'Tcl', 'Stata', 'AppleScript', 'Shell', 'Clojure', 'Scheme', 'ANTLR', 'SPARQL', 'SQL', 'GLSL', 'Elm', 'Dockerfile', 'C++', 'CoffeeScript', 'Common Lisp', 'Elixir', 'Groovy', 'HTML', 'Java', 'JavaScript', 'Markdown', 'PHP', 'Python', 'TypeScript', 'Verilog', 'Visual Basic', 'VHDL', 'Thrift', 'Matlab', 'Yacc', 'Zig', 'XSLT', 'JSON', 'YAML']
CPU times: user 137 ms, sys: 1.65 ms, total: 138 ms
Wall time: 136 ms
```
Compare that to using a standard Python `set()` to achieve the same result which now is much slower, and has to load the entire column into memory:
```python
>>> dset.set_format(None)
>>> %time set(dset["lang"])
TODO
```
Mapping these compute primitives can enable [MapReduce](https://en.wikipedia.org/wiki/MapReduce)-like operations that can be especially useful for processing large datasets. For example, in an effort to better understand the structure of our dataset, we can map a `group_by` operation that groups on the `lang` column and computes the average file size and number of GitHub stars per progamming language.
```python
>>> dset.set_format("arrow")
>>> %time mapped_result = dset.map(lambda table: table.group_by("lang").aggregate([("size", "mean"),("max_stars_count", "mean")]), batched=True, batch_size=100_000, num_proc=10)
CPU times: user 765 ms, sys: 237 ms, total: 1 s
Wall time: 1.35 s
>>> mapped_result
Dataset({
    features: ['size_mean', 'max_stars_count_mean', 'lang'],
    num_rows: 207
})
```
The previous computation would also be possible in `pandas`, provided you have enough memory to materialize the entire table into RAM. Machine learning datasets are typically too big for that, and you would have to use parallel computation frameworks such as [`Dask`](https://docs.dask.org/en/stable/) or [`Spark`](https://huggingface.co/docs/datasets/use_with_spark) to run the same computation.

But for now, let's go ahead and reduce the results of the map operation by calling `group_by` once more with the same aggregations, this time specifying a batch size of `None`: this will take the entire dataset (all 207 rows output by the map operation) and treat them as one batch. The advantage in doing this instead of just calling the `group_by` operation on the underlying Arrow Table, is that the `.map()` operation caches its computations to disk. This means that you only ever have to run any operation once. Any subsequent calls to the same operation will be mmapped from disk. 
```python
>>> reduced_result = mapped_result.map(lambda table: table.group_by("lang").aggregate([("size_mean", "mean"),("max_stars_count_mean", "mean")]), batched=True, batch_size=None)
>>> reduced_result.to_pandas().sort_values("size_mean_mean", ascending=False).head(15)
```

|    |   size_mean_mean |   max_stars_count_mean_mean | lang        |
|---:|-----------------:|----------------------------:|:------------|
| 30 |          63950.9 |                     62.9083 | Mathematica |
| 80 |          18590.4 |                     28.0008 | VHDL        |
|  9 |          17811.9 |                     33.3281 | Isabelle    |
| 68 |          16470.2 |                    118.346  | Common Lisp |
| 83 |          15776.3 |                     55.9424 | Yacc        |
| 71 |          15283.2 |                    294.813  | HTML        |
| 17 |          14399.4 |                     60.4394 | Pascal      |
| 22 |          13275   |                     22.6699 | SAS         |
| 85 |          13040.5 |                    170.301  | XSLT        |
| 55 |          13016.3 |                     34.9036 | Stata       |
| 62 |          12458.2 |                    339.297  | SQL         |
| 29 |          11677.5 |                    112.621  | Maple       |
| 84 |          11385.5 |                    349.611  | Zig         |

```python
python = dset.map(lambda table: table.filter(pc.field("lang") == "Python"), batched=True, batch_size=500_000, num_proc=10)
print(f'{python.num_rows:,}', "rows")
>>> 250,000 rows
CPU times: user 818 ms, sys: 255 ms, total: 1.07 s
Wall time: 3.46 s
```
Compare that to using the native `.filter()` operation in batched mode:
```python
>>> dset.set_format(None)
>>> %time python = dset.filter(lambda batch: [language == "Python" for language in batched["row"]], batched=True, batch_size==500_000, num_proc=10)

```
## Further Resources

- https://blog.djnavarro.net/posts/2022-05-25_arrays-and-tables-in-arrow/
- https://arrow.apache.org/cookbook/
- https://github.com/PacktPublishing/In-Memory-Analytics-with-Apache-Arrow-