---
title: "Introducing the SQL Console on Datasets" 
thumbnail: /blog/assets/sql_console/thumbnail.png
authors:
- user: cfahlgren1
---

Datasets have been exploding and Hugging Face has become the default home for many datasets. 

# Datasets Growth

Each month, the amount of datasets uploaded compounds, and so does the need to query, filter and discover them.

![Dataset Monthly Creations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/dataset_monthly_creations.png)

As the number of datasets has grown, so has the need to query and filter them. 

We are very excited to announce that you can now run SQL queries on your datasets directly in the Hugging Face Hub!

# Introducing the SQL Console on Datasets

On every dataset you should see a new **SQL Console** badge. In one click, you can open a [DuckDB](https://duckdb.org/) SQL Console for the given dataset.

![SQLConsole](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/SQLConsole.gif)

- **No dependencies**: The SQL Console is powered by DuckDB WASM, so you can query your dataset without any dependencies.
- **Full DuckDB Syntax**: DuckDB has [full SQL](https://duckdb.org/docs/sql/introduction.html) syntax support along with many built in functions for regex, lists, JSON, embeddings and more. You'll find, DuckDB syntax is very similar to PostgreSQL.
- **Export Results**: You can export the results of your query to parquet.
- **Shareable**: You can share your query results of public datasets with a link.

# How it works

## Parquet Conversion

To power the dataset viewer on Hugging Face, the first 5GB of every dataset is auto-converted to Parquet (unless it was already a Parquet dataset). Parquet is a columnar data format that is optimized for performance and storage efficiency. 

The beauty of this is that you can run SQL queries on the dataset without needing to download the entire dataset. DuckDB can skip row groups based on filters, utilizing the metadata in the Parquet files. This is done using [DuckDB](https://duckdb.org/) and HTTP requests with byte ranges to the dataset. 

You can learn more about the Parquet format and range requests [here](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format). The DuckDB CLI also supports the `hf://` protocol to read the parquet conversion. 

Here's how it works:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/duckdb_hf_url.png" alt="DuckDB CLI" width="500"/>

## DuckDB WASM 🦆

[DuckDB WASM](https://duckdb.org/docs/api/wasm/overview.html) is the engine that powers the SQL Console. It is an in-process SQL engine that runs on the Web Assembly (WASM). 

This enables it to run entirely in the browser, with no server or backend required. This gives the user the upmost flexibility to query data as they please without any dependencies.

You may be wondering, _"Will it work for big datasets?"_ and the answer is, "Yes!". 

Here's a query of the [OpenCo7/UpVoteWeb](https://huggingface.co/datasets/OpenCo7/UpVoteWeb) dataset which has `12.6M` rows in the Parquet conversion.

![Reddit Movie Suggestions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/reddit-movie-suggestions.png)

You can see we got results for a simple filter query in under 3 seconds. 

While queries will take longer based on the size of the dataset and query complexity, you will be suprised what you can do with the SQL Console.

**Limitations**
- The SQL Console will work for a lot of queries, however, the memory limit is ~3GB, so it is possible to run out of memory and not be able to process the query (_Tip: try to use filters to reduce the amount of data you are querying along with `LIMIT`_).
- While DuckDB WASM is very powerful, it is not 1 to 1 with the DuckDB CLI. For example, it does not yet support the `hf://` protocol to download datasets.

**Try it out!**

You can try out a SQL Console query for [SkunkworksAI/reasoning-0.01](https://huggingface.co/datasets/SkunkworksAI/reasoning-0.01?sql_console=true&sql=--+Find+instructions+with+more+than+10+reasoning+steps%0Aselect+*+from+train%0Awhere+len%28reasoning_chains%29+%3E+10%0Alimit+100&sql_row=43) to see instructions with more than 10 reasoning steps.

## SQL Snippets

DuckDB has a ton of use cases that we are still exploring. We created a [SQL Snippets](https://huggingface.co/spaces/cfahlgren1/sql-snippets) space to showcase what you can do with the SQL Console.

Here are some really interesting use cases we have found:

- [Filtering a function calling dataset for a specific function with regex](https://x.com/qlhoest/status/1835687940376207651)
- [Finding the most popular base models from open-llm-leaderboard](https://x.com/polinaeterna/status/1834601082862842270)
- [Converting an alpaca dataset to a conversational format](https://x.com/calebfahlgren/status/1834674871688704144)
- [Performing similarity search with embeddings](https://x.com/andrejanysa/status/1834253758152269903)
- [Filtering 50k+ rows from a dataset for the highest quality, reasoning instructions](https://x.com/calebfahlgren/status/1835703284943749301)

Remember, it's one click to download your SQL results as a Parquet file and use for your dataset!

We would love to hear what you think of the SQL Console and if you have any feedback, please comment in this [post!](https://huggingface.co/posts/cfahlgren1/845769119345136)

## Example: Converting a dataset from Alpaca to conversations

For finetuning, there are different formats you can use. One common format is conversational. This is where each row is a conversation between a user and the model and consists of multiple turns.

In this example, we will convert an Alpaca dataset to a conversational format. 

Typically, it would be easiest to do this with a script, however, we can also use the SQL Console to do this in 30 seconds. 

<iframe
  src="https://huggingface.co/datasets/yahma/alpaca-cleaned/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>


### SQL Query

```sql
-- Convert Alpaca format to Conversation format
WITH 
source_view AS (
  SELECT * FROM train  -- Change 'train' to your desired view name here
)
SELECT 
  [
    struct_pack(
      "from" := 'user',
      "value" := CASE 
                   WHEN input IS NOT NULL AND input != '' 
                   THEN instruction || '\n\n' || input
                   ELSE instruction
                 END
    ),
    struct_pack(
      "from" := 'assistant',
      "value" := output
    )
  ] AS conversation
FROM source_view
WHERE instruction IS NOT NULL 
AND output IS NOT NULL;
```

The query above is structured to make it easy to swap out the view name. Essentially, we are taking the columns `instruction` and `input` and concatenating them together with a newline. 

We are also adding a `from` field to the conversation to indicate that the message is from the assistant. We use `struct_pack` to create a new STRUCT row for each conversation.

DuckDB has some great documentation on the `STRUCT` [Data Type](https://duckdb.org/docs/sql/data_types/struct.html) and [Functions](https://duckdb.org/docs/sql/functions/struct.html).

![Alpaca to Conversation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/alpaca-to-conversation.png)

Once we have the results, we can download the results as a Parquet file. You can see what the final output looks like below.

<iframe
  src="https://huggingface.co/datasets/cfahlgren1/alpaca-conversational/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

### Resources

- [DuckDB WASM](https://duckdb.org/docs/api/wasm/overview.html)
- [DuckDB Syntax](https://duckdb.org/docs/sql/introduction.html)
- [DuckDB WASM Paper](https://www.vldb.org/pvldb/vol15/p3574-kohn.pdf)
- [Intro to Parquet Format](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format)
- [Hugging Face + DuckDB](https://huggingface.co/docs/hub/en/datasets-duckdb)
- [SQL Snippets Space](https://huggingface.co/spaces/cfahlgren1/sql-snippets)