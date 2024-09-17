---
title: "Introducing the SQL Console on Datasets" 
thumbnail: /blog/assets/sql_console/thumbnail.png
authors:
- user: cfahlgren1
---

Datasets have been exploding and Hugging Face has become the default home for many datasets. 

# Datasets Growth

Each month, as the amount of datasets uploaded compounds, and so does the need to query, filter and discover them.

![Dataset Monthly Creations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/dataset_monthly_creations.png) 
_Datasets created on Hugging Face Hub each month_

We are very excited to announce that you can now run SQL queries on your datasets directly in the Hugging Face Hub!

# Introducing the SQL Console for Datasets

On every dataset you should see a new **SQL Console** badge. In one click, you can open a [DuckDB](https://duckdb.org/) SQL Console for the given dataset.

<figure class="image flex flex-col items-center text-center m-0 w-full">
   <video
      alt="SQL Console Demo"
      autoplay loop autobuffer muted playsinline
    >
    <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/Magpie-Ultra-Demo-SQL-Console.mp4" type="video/mp4">
  </video>
  <figcaption class="text-center text-sm italic">Querying the Magpie-Ultra dataset for excellent, high quality reasoning instructions.</figcaption>
</figure>

All the work is done in the browser and the console comes with a few neat features:

- **100% Local**: The SQL Console is powered by DuckDB WASM, so you can query your dataset without any dependencies.
- **Full DuckDB Syntax**: DuckDB has [full SQL](https://duckdb.org/docs/sql/introduction.html) syntax support along with many built in functions for regex, lists, JSON, embeddings and more. You'll find, DuckDB syntax is very similar to PostgreSQL.
- **Export Results**: You can export the results of your query to parquet.
- **Shareable**: You can share your query results of public datasets with a link.

# How it works

## Parquet Conversion

To power the dataset viewer on Hugging Face, the first 5GB of every dataset is auto-converted to Parquet (unless it was already a Parquet dataset, then you have the full dataset). Parquet is a columnar data format that is optimized for performance and storage efficiency. You can find more information about the Parquet conversion process in the [Parquet List API documentation](https://huggingface.co/docs/dataset-viewer/en/parquet).

Using this parquet conversion, the SQL Console creates views for you to query based on your dataset splits and configs. 

## DuckDB WASM 🦆

[DuckDB WASM](https://duckdb.org/docs/api/wasm/overview.html) is the engine that powers the SQL Console. It is an in-process database engine that runs on Web Assembly in the browser. No server or backend needed.

By being soley in the browser, it gives the user the upmost flexibility to query data as they please without any dependencies. It also makes it really simple to share reproducible results with a simple link.

You may be wondering, _"Will it work for big datasets?"_ and the answer is, "Yes!". 

Here's a query of the [OpenCo7/UpVoteWeb](https://huggingface.co/datasets/OpenCo7/UpVoteWeb) dataset which has `12.6M` rows in the Parquet conversion.

![Reddit Movie Suggestions](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/reddit-movie-suggestions.png)

You can see we received results for a simple filter query in under 3 seconds. 

While queries will take longer based on the size of the dataset and query complexity, you will be suprised about how much you can do with the SQL Console. 

As with any technology, there are limitations.
- The SQL Console will work for a lot of queries, however, the memory limit is ~3GB, so it is possible to run out of memory and not be able to process the query (_Tip: try to use filters to reduce the amount of data you are querying along with `LIMIT`_).
- While DuckDB WASM is very powerful, it is not fully feature parity with DuckDB. For example, DuckDB WASM does not yet support the [`hf://` protocol to query datasets](https://github.com/duckdb/duckdb-wasm/discussions/1858).

## Example: Converting a dataset from Alpaca to conversations

Now that we've introduced the SQL Console, let's explore a practical example. When fine-tuning a Large Language Model (LLM), you often need to work with different data formats. One particularly popular format is the conversational format, where each row represents a multi-turn dialogue between a user and the model. The SQL Console can help us transform data into this format efficiently. Let's see how we can convert an Alpaca dataset to a conversational format using SQL.

In this example, we will convert an Alpaca dataset to a conversational format. 

Typically, it would be easiest to do this with a Python script, however, we can also use the SQL Console to do this in less than30 seconds. 

<iframe
  src="https://huggingface.co/datasets/yahma/alpaca-cleaned/embed/viewer/default/train?sql=--+Convert+Alpaca+format+to+Conversation+format%0AWITH+%0Asource_view+AS+%28%0A++SELECT+*+FROM+train++--+Change+%27train%27+to+your+desired+view+name+here%0A%29%0ASELECT+%0A++%5B%0A++++struct_pack%28%0A++++++%22from%22+%3A%3D+%27user%27%2C%0A++++++%22value%22+%3A%3D+CASE+%0A+++++++++++++++++++WHEN+input+IS+NOT+NULL+AND+input+%21%3D+%27%27+%0A+++++++++++++++++++THEN+instruction+%7C%7C+%27%5Cn%5Cn%27+%7C%7C+input%0A+++++++++++++++++++ELSE+instruction%0A+++++++++++++++++END%0A++++%29%2C%0A++++struct_pack%28%0A++++++%22from%22+%3A%3D+%27assistant%27%2C%0A++++++%22value%22+%3A%3D+output%0A++++%29%0A++%5D+AS+conversation%0AFROM+source_view%0AWHERE+instruction+IS+NOT+NULL+%0AAND+output+IS+NOT+NULL%3B"
  frameborder="0"
  width="100%"
  height="800px"
></iframe>

In the dataset above, click on the **SQL Console** badge to open the SQL Console. You should see the query below automatically populated.

When you are ready, click the **Run Query** button to execute the query. 

### SQL

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

In the query we use the `struct_pack` function to create a new STRUCT row for each conversation.

DuckDB has great documentation on the `STRUCT` [Data Type](https://duckdb.org/docs/sql/data_types/struct.html) and [Functions](https://duckdb.org/docs/sql/functions/struct.html). You'll find many datasets contain columns with JSON data. DuckDB provides functions to easily parse and query these columns.

![Alpaca to Conversation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/sql_console/alpaca-to-conversation.png)

Once we have the results, we can download the results as a Parquet file. You can see what the final output looks like below.

<iframe
  src="https://huggingface.co/datasets/cfahlgren1/alpaca-conversational/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

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

### Resources

- [DuckDB WASM](https://duckdb.org/docs/api/wasm/overview.html)
- [DuckDB Syntax](https://duckdb.org/docs/sql/introduction.html)
- [DuckDB WASM Paper](https://www.vldb.org/pvldb/vol15/p3574-kohn.pdf)
- [Intro to Parquet Format](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format)
- [Hugging Face + DuckDB](https://huggingface.co/docs/hub/en/datasets-duckdb)
- [SQL Snippets Space](https://huggingface.co/spaces/cfahlgren1/sql-snippets)