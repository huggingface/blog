---
title: "How we leveraged distilabel to create an Argilla 2.0 Chatbot"
thumbnail: /blog/assets/argilla-chatbot/thumbnail.png
authors:
- user: plaguss
- user: gabrielmbmb
- user: sdiazlor
- user: osanseviero
- user: dvilasuero
---

# How we leveraged distilabel to create an Argilla 2.0 Chatbot

## TL;DR

Discover how to build a Chatbot for a tool of your choice ([Argilla 2.0](https://github.com/argilla-io/argilla) in this case) that can understand technical documentation and chat with users about it. 

In this article, we'll show you how to leverage [distilabel](https://github.com/argilla-io/distilabel) and fine-tune a domain-specific embedding model to create a conversational model that's both accurate and engaging. 

This article outlines the process of creating a Chatbot for Argilla 2.0. We will:

* create a synthetic dataset from the technical documentation to fine-tune a domain-specific embedding model,
* create a vector database to store and retrieve the documentation and
* deploy the final Chatbot to a Hugging Face Space allowing users to interact with it, storing the interactions in Argilla for continuous evaluation and improvement.

Click the next image to go to the app.

<a href="https://huggingface.co/spaces/plaguss/argilla-sdk-chatbot-space" rel="some text">![argilla-sdk-chatbot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-chatbot/chatbot.png)</a>


## Table of Contents

- [Generating Synthetic Data for Fine-Tuning a domain-specific Embedding Models](#generating-synthetic-data-for-fine-tuning-domain-specific-embedding-models)
- [Downloading and chunking data](#downloading-and-chunking-data)
- [Generating synthetic data for our embedding model using distilabel](#generating-synthetic-data-for-our-embedding-model-using-distilabel)
- [Explore the datasets in Argilla](#explore-the-datasets-in-argilla)
    - [An Argilla dataset with chunks of technical documentation](#an-argilla-dataset-with-chunks-of-technical-documentation)
    - [An Argilla dataset with triplets to fine tune an embedding model](#an-argilla-dataset-with-triplets-to-fine-tune-an-embedding-model)
    - [An Argilla dataset to track the chatbot conversations](#an-argilla-dataset-to-track-the-chatbot-conversations)
- [Fine-Tune the embedding model](#fine-tune-the-embedding-model)
    - [Prepare the embedding dataset](#prepare-the-embedding-dataset)
    - [Load the baseline model](#load-the-baseline-model)
    - [Define the loss function](#define-the-loss-function)
    - [Define the training strategy](#define-the-training-strategy)
    - [Train and save the final model](#train-and-save-the-final-model)
- [The vector database](#the-vector-database)
- [Connect to the database](#connect-to-the-database)
    - [Instantiate the fine-tuned model](#instantiate-the-fine-tuned-model)
- [Create the table with the documentation chunks](#create-the-table-with-the-documentation-chunks)
    - [Populate the table](#populate-the-table)
    - [Store the database in the Hugging Face Hub](#store-the-database-in-the-hugging-face-hub)
- [Creating our ChatBot](#creating-our-chatbot)
- [The Gradio App](#the-gradio-app)
- [Deploy the ChatBot app on Hugging Face Spaces](#deploy-the-chatbot-app-on-hugging-face-spaces)
- [Playing around with our ChatBot](#playing-around-with-our-chatbot)
- [Next steps](#next-steps)

## Generating Synthetic Data for Fine-Tuning Custom Embedding Models

Need a quick recap on RAG? Brush up on the basics with this handy [intro notebook](https://huggingface.co/learn/cookbook/en/rag_zephyr_langchain#simple-rag-for-github-issues-using-hugging-face-zephyr-and-langchain). We'll wait for you to get up to speed!

### Downloading and chunking data

Chunking data means dividing your text data into manageable chunks of approximately 256 tokens each (chunk size used in RAG later).

Let's dive into the first step: processing the documentation of your target repository. To simplify this task, you can leverage libraries like [llama-index](https://docs.llamaindex.ai/en/stable/examples/data_connectors/GithubRepositoryReaderDemo/) to read the repository contents and parse the markdown files. Specifically, langchain offers useful tools like [MarkdownTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/markdown_header_metadata/) and `llama-index` provides [MarkdownNodeParser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/?h=markdown#markdownnodeparser) to help you extract the necessary information. If you prefer a more streamlined approach, consider using the [corpus-creator](https://huggingface.co/spaces/davanstrien/corpus-creator) app from [`davanstrien`](https://huggingface.co/davanstrien).

To make things easier and more efficient, we've developed a custom Python script that does the heavy lifting for you. You can find it in our repository [here](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/docs_dataset.py).

This script automates the process of retrieving documentation from a GitHub repository and storing it as a dataset on the Hugging Face Hub. And the best part? It's incredibly easy to use! Let's see how we can run it:

```bash
python docs_dataset.py \
    "argilla-io/argilla-python" \
    --dataset-name "plaguss/argilla_sdk_docs_raw_unstructured"
```

<!-- There are some additional arguments you can use, but the required ones are the GitHub path to the repository where the docs are located and the dataset ID for the Hugging Face Hub. The script will download the docs (located at `/docs` by default, but it can be changed as shown in the following snippet) to your local directory, extract all the markdown files, chunk them, and push the dataset to the Hugging Face Hub. The core logic can be summarized by the following snippet: -->

While the script is easy to use, you can further tailor it to your needs by utilizing additional arguments. However, there are two essential inputs you'll need to provide:

- The GitHub path to the repository where your documentation is stored

- The dataset ID for the Hugging Face Hub, where your dataset will be stored

Once you've provided these required arguments, the script will take care of the rest. Here's what happens behind the scenes:

- The script downloads the documentation from the specified GitHub repository to your local directory. By default, it looks for docs in the `/docs` directory by default, but you can change this by specifying a different path.

- It extracts all the markdown files from the downloaded documentation.

- Chunks the extracted markdown files into manageable pieces.

- Finally, it pushes the prepared dataset to the Hugging Face Hub, making it ready for use.

To give you a better understanding of the script's inner workings, here's a code snippet that summarizes the core logic:

```python
# The function definitions are omitted for brevity, visit the script for more info!
from github import Github

gh = Github()
repo = gh.get_repo("repo_name")

# Download the folder
download_folder(repo, "/folder/with/docs", "dir/to/download/docs") 

# Extract the markdown files from the downloaded folder with the documentation from the GitHub repository
md_files = list(docs_path.glob("**/*.md"))

# Loop to iterate over the files and generate chunks from the text pieces
data = create_chunks(md_files)

# Create a dataset to push it to the hub
create_dataset(data, repo_name="name/of/the/dataset")
```

The script includes short functions to download the documentation, create chunks from the markdown files, and create the dataset. Including more functionalities or implementing a more complex chunking strategy should be straightforward.

You can take a look at the available arguments:

<details close>
<summary>Click to see docs_dataset.py help message</summary>

```bash
$ python docs_dataset.py -h
usage: docs_dataset.py [-h] [--dataset-name DATASET_NAME] [--docs_folder DOCS_FOLDER] [--output_dir OUTPUT_DIR] [--private | --no-private] repo [repo ...]

Download the docs from a github repository and generate a dataset from the markdown files. The dataset will be pushed to the hub.

positional arguments:
  repo                  Name of the repository in the hub. For example 'argilla-io/argilla-python'.

options:
  -h, --help            show this help message and exit
  --dataset-name DATASET_NAME
                        Name to give to the new dataset. For example 'my-name/argilla_sdk_docs_raw'.
  --docs_folder DOCS_FOLDER
                        Name of the docs folder in the repo, defaults to 'docs'.
  --output_dir OUTPUT_DIR
                        Path to save the downloaded files from the repo (optional)
  --private, --no-private
                        Whether to keep the repository private or not. Defaults to False.
```

</details>


### Generating synthetic data for our embedding model using distilabel

We will generate synthetic questions from our documentation that can be answered by every chunk of documentation. We will also generate hard negative examples by generating unrelated questions that can be easily distinguishable. We can use the questions, hard negatives, and docs to build the triples for the fine-tuning dataset.

The full pipeline script can be seen at [`pipeline_docs_queries.py`](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/pipeline_docs_queries.py) in the reference repository, but let's go over the different steps:

1. `load_data`:

The first step in our journey is to acquire the dataset that houses the valuable documentation chunks. Upon closer inspection, we notice that the column containing these chunks is aptly named `chunks`. However, for our model to function seamlessly, we need to assign a new identity to this column. Specifically, we want to rename it to `anchor`, as this is the input our subsequent steps will be expecting. We'll make use of `output_mappings` to do this column transformation for us:

```python
load_data = LoadDataFromHub(
    name="load_data",
    repo_id="plaguss/argilla_sdk_docs_raw_unstructured",
    output_mappings={"chunks": "anchor"},
    batch_size=10,
)
```

2. `generate_sentence_pair`

Now, we arrive at the most fascinating part of our process, transforming the documentation pieces into synthetic queries. This is where the [`GenerateSentencePair`](https://distilabel.argilla.io/latest/components-gallery/tasks/generatesentencepair/) task takes center stage. This powerful task offers a wide range of possibilities for generating high-quality sentence pairs. We encourage you to explore its documentation to unlock its full potential.

In our specific use case, we'll harness the capabilities of [`GenerateSentencePair`](https://distilabel.argilla.io/latest/components-gallery/tasks/generatesentencepair/) to craft synthetic queries that will ultimately enhance our model's performance. Let's dive deeper into how we'll configure this task to achieve our goals.

```python
llm = InferenceEndpointsLLM(
    model_id="meta-llama/Meta-Llama-3-70B-Instruct",
    tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
)

generate_sentence_pair = GenerateSentencePair(
    name="generate_sentence_pair",
    triplet=True,  # Generate positive and negative
    action="query",
    context="The generated sentence has to be related with Argilla, a data annotation tool for AI engineers and domain experts.",
    llm=llm,
    input_batch_size=10,
    output_mappings={"model_name": "model_name_query"},
)
```

Let's break down the code snippet above.

By setting `triplet=True`, we're instructing the task to produce a series of triplets, comprising an anchor, a positive sentence, and a negative sentence. This format is perfectly suited for fine-tuning, as explained in the Sentence Transformers library's [training overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html).

The `action="query"` parameter is a crucial aspect of this task, as it directs the LLM to generate queries for the positive sentences. This is where the magic happens, and our documentation chunks are transformed into meaningful queries.

To further assist the model, we've included the `context` argument. This provides additional information to the LLM when the anchor sentence lacks sufficient context, which is often the case with brief documentation chunks.

Finally, we've chosen to harness the power of the `meta-llama/Meta-Llama-3-70B-Instruct` model, via the [`InferenceEndpointsLLM`](https://distilabel.argilla.io/latest/components-gallery/llms/inferenceendpointsllm/) component. This selection enables us to tap into the model's capabilities, generating high-quality synthetic queries that will ultimately enhance our model's performance.


3. `multiply_queries`

Using the `GenerateSentencePair` step, we obtained as many examples for training as chunks we had, 251 in this case. However, we recognize that this might not be sufficient to fine-tune a custom model that can accurately capture the nuances of our specific use case.

To overcome this limitation, we'll employ another LLM to generate additional queries. This will allow us to increase the size of our training dataset, providing our model with a richer foundation for learning.

This brings us to the next step in our pipeline: `MultipleQueries`, a custom `Task` that we've crafted to further augment our dataset.

```python
multiply_queries = MultipleQueries(
    name="multiply_queries",
    num_queries=3,
    system_prompt=(
        "You are an AI assistant helping to generate diverse examples. Ensure the "
        "generated queries are all in separated lines and preceded by a dash. "
        "Do not generate anything else or introduce the task."
    ),
    llm=llm,
    input_batch_size=10,
    input_mappings={"query": "positive"},
    output_mappings={"model_name": "model_name_query_multiplied"},
)
```

Now, let's delve into the configuration of our custom `Task`, designed to amplify our training dataset. The linchpin of this task is the `num_queries` parameter, set to 3 in this instance. This means we'll generate three additional "positive" queries for each example, effectively quadrupling our dataset size, assuming some examples may not succeed.

To ensure the Large Language Model (LLM) stays on track, we've crafted a system_prompt that provides clear guidance on our instructions. Given the strength of the chosen model and the simplicity of our examples, we didn't need to employ structured generation techniques. However, this could be a valuable approach in more complex scenarios.

Curious about the inner workings of our custom `Task`? Click the dropdown below to explore the full definition:

<details close>
<summary>MultipleQueries definition</summary>
<br>

```python
multiply_queries_template = (
    "Given the following query:\n{original}\nGenerate {num_queries} similar queries by varying "
    "the tone and the phrases slightly. "
    "Ensure the generated queries are coherent with the original reference and relevant to the context of data annotation "
    "and AI dataset development."
)

class MultipleQueries(Task):
    system_prompt: Optional[str] = None
    num_queries: int = 1

    @property
    def inputs(self) -> List[str]:
        return ["query"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        prompt = [
            {
                "role": "user",
                "content": multiply_queries_template.format(
                    original=input["query"],
                    num_queries=self.num_queries
                ),
            },
        ]
        if self.system_prompt:
            prompt.insert(0, {"role": "system", "content": self.system_prompt})
        return prompt

    @property
    def outputs(self) -> List[str]:
        return ["queries", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        queries = output.split("- ")
        if len(queries) > self.num_queries:
            queries = queries[1:]
        queries = [q.strip() for q in queries]
        return {"queries": queries}
```

</details><p>

4) `merge_columns`

As we approach the final stages of our pipeline, our focus shifts to data processing. Our ultimate goal is to create a refined dataset, comprising rows of triplets suited for fine-tuning. However, after generating multiple queries, our dataset now contains two distinct columns: `positive` and `queries`. The `positive` column holds the original query as a single string, while the `queries` column stores a list of strings, representing the additional queries generated for the same entity.

To merge these two columns into a single, cohesive list, we'll employ the [`MergeColumns`](https://distilabel.argilla.io/dev/components-gallery/steps/mergecolumns/) step. This will enable us to combine the original query with the generated queries, creating a unified:

```python
merge_columns = MergeColumns(
    name="merge_columns",
    columns=["positive", "queries"],
    output_column="positive"
)
```

5) `expand_columns`

Lastly, we use [`ExpandColumns`](https://distilabel.argilla.io/dev/components-gallery/steps/expandcolumns/) to move the previous column of positive to different lines. As a result, each `positive` query will occupy a separate line, while the `anchor` and `negative` columns will be replicated to match the expanded positive queries. This data manipulation will yield a dataset with the ideal structure for fine-tuning:

```python
expand_columns = ExpandColumns(columns=["positive"])
```

Click the dropdown to see the full pipeline definition:

<details close>
<summary>Distilabel Pipeline</summary>
<br>

```python
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateSentencePair
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import ChatType
from distilabel.steps import ExpandColumns, CombineKeys


multiply_queries_template = (
    "Given the following query:\n{original}\nGenerate {num_queries} similar queries by varying "
    "the tone and the phrases slightly. "
    "Ensure the generated queries are coherent with the original reference and relevant to the context of data annotation "
    "and AI dataset development."
)

class MultipleQueries(Task):
    system_prompt: Optional[str] = None
    num_queries: int = 1

    @property
    def inputs(self) -> List[str]:
        return ["query"]

    def format_input(self, input: Dict[str, Any]) -> ChatType:
        prompt = [
            {
                "role": "user",
                "content": multiply_queries_template.format(
                    original=input["query"],
                    num_queries=self.num_queries
                ),
            },
        ]
        if self.system_prompt:
            prompt.insert(0, {"role": "system", "content": self.system_prompt})
        return prompt

    @property
    def outputs(self) -> List[str]:
        return ["queries", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        queries = output.split("- ")
        if len(queries) > self.num_queries:
            queries = queries[1:]
        queries = [q.strip() for q in queries]
        return {"queries": queries}


with Pipeline(
    name="embedding-queries",
    description="Generate queries to train a sentence embedding model."
) as pipeline:
    load_data = LoadDataFromHub(
        name="load_data",
        repo_id="plaguss/argilla_sdk_docs_raw_unstructured",
        output_mappings={"chunks": "anchor"},
        batch_size=10,
    )

    llm = InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
    )

    generate_sentence_pair = GenerateSentencePair(
        name="generate_sentence_pair",
        triplet=True,  # Generate positive and negative
        action="query",
        context="The generated sentence has to be related with Argilla, a data annotation tool for AI engineers and domain experts.",
        llm=llm,
        input_batch_size=10,
        output_mappings={"model_name": "model_name_query"},
    )

    multiply_queries = MultipleQueries(
        name="multiply_queries",
        num_queries=3,
        system_prompt=(
            "You are an AI assistant helping to generate diverse examples. Ensure the "
            "generated queries are all in separated lines and preceded by a dash. "
            "Do not generate anything else or introduce the task."
        ),
        llm=llm,
        input_batch_size=10,
        input_mappings={"query": "positive"},
        output_mappings={"model_name": "model_name_query_multiplied"},
    )

    merge_columns = MergeColumns(
        name="merge_columns",
        columns=["positive", "queries"],
        output_column="positive"
    )

    expand_columns = ExpandColumns(
        columns=["positive"],
    )

    (
        load_data
        >> generate_sentence_pair
        >> multiply_queries
        >> merge_columns
        >> expand_columns
    )


if __name__ == "__main__":

    pipeline_parameters = {
        "generate_sentence_pair": {
            "llm": {
                "generation_kwargs": {
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                }
            }
        },
        "multiply_queries": {
            "llm": {
                "generation_kwargs": {
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                }
            }
        }
    }

    distiset = pipeline.run(
        parameters=pipeline_parameters
    )
    distiset.push_to_hub("plaguss/argilla_sdk_docs_queries")
```

</details>

### Explore the datasets in Argilla

Now that we've generated our datasets, it's time to dive deeper and refine them as needed using Argilla. To get started, take a look at our [argilla_datasets.ipynb](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/argilla_datasets.ipynb) notebook, which provides a step-by-step guide on how to upload your datasets to Argilla.

If you haven't set up an Argilla instance yet, don't worry! Follow our easy-to-follow guide in the [docs](https://argilla-io.github.io/argilla/latest/getting_started/quickstart/#run-the-argilla-server) to create a Hugging Face Space with Argilla. Once you've got your Space up and running, simply connect to it by updating the `api_url` to point to your Space:


```python
import argilla as rg

client = rg.Argilla(
    api_url="https://plaguss-argilla-sdk-chatbot.hf.space",
    api_key="YOUR_API_KEY"
)
```

#### An Argilla dataset with chunks of technical documentation

With your Argilla instance up and running, let's move on to the next step: configuring the `Settings` for your dataset. The good news is that the default `Settings` we'll create should work seamlessly for your specific use case, with no need for further adjustments:

```python
settings = rg.Settings(
    guidelines="Review the chunks of docs.",
    fields=[
        rg.TextField(
            name="filename",
            title="Filename where this chunk was extracted from",
            use_markdown=False,
        ),
        rg.TextField(
            name="chunk",
            title="Chunk from the documentation",
            use_markdown=False,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="good_chunk",
            title="Does this chunk contain relevant information?",
            labels=["yes", "no"],
        )
    ],
)
```

Let's take a closer look at the dataset structure we've created. We'll examine the `filename` and `chunk` fields, which contain the parsed filename and the generated chunks, respectively. To further enhance our dataset, we can define a simple label question, `good_chunk`, which allows us to manually label each chunk as useful or not. This human-in-the-loop approach enables us to refine our automated generation process. With these essential elements in place, we're now ready to create our dataset:

```python
dataset = rg.Dataset(
    name="argilla_sdk_docs_raw_unstructured",
    settings=settings,
    client=client,
)
dataset.create()
```

Now, let's retrieve the dataset we created earlier from the Hugging Face Hub. Recall the dataset we generated in the [chunking data section](#downloading-and-chunking-data)? We'll download that dataset and extract the essential columns we need to move forward:

```python
from datasets import load_dataset

data = (
    load_dataset("plaguss/argilla_sdk_docs_raw_unstructured", split="train")
    .select_columns(["filename", "chunks"])
    .to_list()
)
```

We've reached the final milestone! To bring everything together, let's log the records to Argilla. This will allow us to visualize our dataset in the Argilla interface, providing a clear and intuitive way to explore and interact with our data:

```python
dataset.records.log(records=data, mapping={"filename": "filename", "chunks": "chunk"})
```

These are the kind of examples you could expect to see:

![argilla-img-1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-chatbot/argilla-img-1.png)

#### An Argilla dataset with triplets to fine-tune an embedding model

Now, we can repeat the process with the dataset ready for fine-tuning we generated in the [previous section](#generating-synthetic-data-for–our-embedding-model:-distilabel-to-the-rescue). 
Fortunately, the process is straightforward: simply download the relevant dataset and upload it to Argilla with its designated name. For a detailed walkthrough, refer to the Jupyter notebook, which contains all the necessary instructions:

```python
settings = rg.Settings(
    guidelines="Review the chunks of docs.",
    fields=[
        rg.TextField(
            name="anchor",
            title="Anchor (Chunk from the documentation).",
            use_markdown=False,
        ),
        rg.TextField(
            name="positive",
            title="Positive sentence that queries the anchor.",
            use_markdown=False,
        ),
        rg.TextField(
            name="negative",
            title="Negative sentence that may use similar words but has content unrelated to the anchor.",
            use_markdown=False,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="is_positive_relevant",
            title="Is the positive query relevant?",
            labels=["yes", "no"],
        ),
        rg.LabelQuestion(
            name="is_negative_irrelevant",
            title="Is the negative query irrelevant?",
            labels=["yes", "no"],
        )
    ],
)
```

Let's take a closer look at the structure of our dataset, which consists of three essential [`TextFields`](https://argilla-io.github.io/argilla/latest/reference/argilla/settings/fields/?h=textfield): `anchor`, `positive`, and `negative`. The `anchor` field represents the chunk of text itself, while the `positive` field contains a query that can be answered using the anchor text as a reference. In contrast, the `negative` field holds an unrelated query that serves as a negative example in the triplet. The positive and negative questions play a crucial role in helping our model distinguish between these examples and learn effective embeddings.

An example can be seen in the following image:

![argilla-img-2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-chatbot/argilla-img-2.png)

The dataset settings we've established so far have been focused on exploring our dataset, but we can take it a step further. By customizing these settings, we can identify and correct incorrect examples, refine the quality of generated questions, and iteratively improve our dataset to better suit our needs.

#### An Argilla dataset to track the chatbot conversations

Now, let's create our final dataset, which will be dedicated to tracking user interactions with our chatbot. *Note*: You may want to revisit this section after completing the Gradio app, as it will provide a more comprehensive understanding of the context. For now, let's take a look at the `Settings` for this dataset:

```python
settings_chatbot_interactions = rg.Settings(
    guidelines="Review the user interactions with the chatbot.",
    fields=[
        rg.TextField(
            name="instruction",
            title="User instruction",
            use_markdown=True,
        ),
        rg.TextField(
            name="response",
            title="Bot response",
            use_markdown=True,
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="is_response_correct",
            title="Is the response correct?",
            labels=["yes", "no"],
        ),
        rg.LabelQuestion(
            name="out_of_guardrails",
            title="Did the model answered something out of the ordinary?",
            description="If the model answered something unrelated to Argilla SDK",
            labels=["yes", "no"],
        ),
        rg.TextQuestion(
            name="feedback",
            title="Let any feedback here",
            description="This field should be used to report any feedback that can be useful",
            required=False
        ),
    ],
    metadata=[
        rg.TermsMetadataProperty(
            name="conv_id",
            title="Conversation ID",
        ),
        rg.IntegerMetadataProperty(
            name="turn",
            min=0,
            max=100,
            title="Conversation Turn",
        )
    ]
)
```

In this dataset, we'll define two essential fields: `instruction` and `response`. The `instruction` field will store the initial query, and if the conversation is extended, it will contain the entire conversation history up to that point. The `response` field, on the other hand, will hold the chatbot's most recent response. To facilitate evaluation and feedback, we'll include three questions: one to assess the correctness of the response, another to determine if the model strayed off-topic, and an optional field for users to provide feedback on the response. Additionally, we'll include two metadata properties to enable filtering and analysis of the conversations: a unique conversation ID and the turn number within the conversation.

An example can be seen in the following image:

![argilla-img-3](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-chatbot/argilla-img-3.png)

Once our chatbot has garnered significant user engagement, this dataset can serve as a valuable resource to refine and enhance our model, allowing us to iterate and improve its performance based on real-world interactions.

### Fine-Tune the embedding model

Now that our custom embedding model dataset is prepared, it's time to dive into the training process.

To guide us through this step, we'll be referencing the [`train_embedding.ipynb`](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/train_embedding.ipynb) notebook, which draws inspiration from Philipp Schmid's [blog post](https://www.philschmid.de/fine-tune-embedding-model-for-rag) on fine-tuning embedding models for RAG. While the blog post provides a comprehensive overview of the process, we'll focus on the key differences and nuances specific to our use case.

For a deeper understanding of the underlying decisions and a detailed walkthrough, be sure to check out the original blog post and review the notebook for a step-by-step explanation.

#### Prepare the embedding dataset

We'll begin by downloading the dataset and selecting the essential columns, which conveniently already align with the naming conventions expected by Sentence Transformers. Next, we'll add a unique id column to each sample and split the dataset into training and testing sets, allocating 90% for training and 10% for testing. Finally, we'll convert the formatted dataset into a JSON file, ready to be fed into the trainer for model fine-tuning:

```python
from datasets import load_dataset

# Load dataset from the hub
dataset = (
    load_dataset("plaguss/argilla_sdk_docs_queries", split="train")
    .select_columns(["anchor", "positive", "negative"])  # Select the relevant columns
    .add_column("id", range(len(dataset)))               # Add an id column to the dataset
    .train_test_split(test_size=0.1)                     # split dataset into a 10% test set
)
 
# Save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")
```

#### Load the baseline model

With our dataset in place, we can now load the baseline model that will serve as the foundation for our fine-tuning process. We'll be using the same model employed in the reference blog post, ensuring a consistent starting point for our custom embedding model development:

```python
from sentence_transformers import SentenceTransformerModelCardData, SentenceTransformer
 
model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="BGE base ArgillaSDK Matryoshka",
    ),
)
```

#### Define the loss function

Given the structure of our dataset, we'll leverage the `TripletLoss` function, which is better suited to handle our `(anchor-positive-negative)` triplets. Additionally, we'll combine it with the `MatryoshkaLoss`, a powerful loss function that has shown promising results (for a deeper dive into `MatryoshkaLoss`, check out [this article](https://huggingface.co/blog/matryoshka)):

```python
from sentence_transformers.losses import MatryoshkaLoss, TripletLoss
 
inner_train_loss = TripletLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=[768, 512, 256, 128, 64]
)
```

#### Define the training strategy

Now that we have our baseline model and loss function in place, it's time to define the training arguments that will guide the fine-tuning process. Since this work was done on an Apple M2 Pro, we need to make some adjustments to ensure a smooth training experience.

To accommodate the limited resources of our machine, we'll reduce the `per_device_train_batch_size` and `per_device_eval_batch_size` compared to the original blog post. Additionally, we'll need to remove the `tf32` and `bf16` precision options, as they're not supported on this device. Furthermore, we'll swap out the `adamw_torch_fused` optimizer, which can be used in a Google Colab notebook for faster training. By making these modifications, we'll be able to fine-tune our model:

```python
from sentence_transformers import SentenceTransformerTrainingArguments
  
# Define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="bge-base-argilla-sdk-matryoshka", # output directory and hugging face model ID
    num_train_epochs=3,                           # number of epochs
    per_device_train_batch_size=8,                # train batch size
    gradient_accumulation_steps=4,                # for a global batch size of 512
    per_device_eval_batch_size=4,                 # evaluation batch size
    warmup_ratio=0.1,                             # warmup ratio
    learning_rate=2e-5,                           # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",                   # use constant learning rate scheduler
    eval_strategy="epoch",                        # evaluate after each epoch
    save_strategy="epoch",                        # save after each epoch
    logging_steps=5,                              # log every 10 steps
    save_total_limit=1,                           # save only the last 3 models
    load_best_model_at_end=True,                  # load the best model when training ends
    metric_for_best_model="eval_dim_512_cosine_ndcg@10",  # optimizing for the best ndcg@10 score for the 512 dimension
)
```

#### Train and save the final model

```python
from sentence_transformers import SentenceTransformerTrainer
 
trainer = SentenceTransformerTrainer(
    model=model,    # bg-base-en-v1
    args=args,      # training arguments
    train_dataset=train_dataset.select_columns(
        ["anchor", "positive", "negative"]
    ),  # training dataset
    loss=train_loss,
    evaluator=evaluator,
)

# Start training, the model will be automatically saved to the hub and the output directory
trainer.train()
 
# Save the best model
trainer.save_model()
 
# Push model to hub
trainer.model.push_to_hub("bge-base-argilla-sdk-matryoshka")
```

And that's it! We can take a look at the new model: [plaguss/bge-base-argilla-sdk-matryoshka](https://huggingface.co/plaguss/bge-base-argilla-sdk-matryoshka). Take a closer look at the dataset card, which is packed with valuable insights and information about our model.

But that's not all! In the next section, we'll put our model to the test and see it in action.

## The vector database

We've made significant progress so far, creating a dataset and fine-tuning a model for our RAG chatbot. Now, it's time to construct the vector database that will empower our chatbot to store and retrieve relevant information efficiently.

When it comes to choosing a vector database, there are numerous alternatives available. To keep things simple and straightforward, we'll be using [lancedb](https://lancedb.github.io/lancedb/), a lightweight, embedded database that doesn't require a server, similar to SQLite. As we'll see, lancedb allows us to create a simple file to store our embeddings, making it easy to move around and retrieve data quickly, which is perfect for our use case.

To follow along, please refer to the accompanying notebook: [`vector_db.ipynb`](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/vector_db.ipynb). In this notebook, we'll delve into the details of building and utilizing our vector database.

### Connect to the database

After installing the dependencies, let's instantiate the database:

```python
import lancedb

# Create a database locally called `lancedb`
db = lancedb.connect("./lancedb")
```

As we execute the code, a new folder should materialize in our current working directory, signaling the successful creation of our vector database.

#### Instantiate the fine-tuned model

Now that our vector database is set up, it's time to load our fine-tuned model. We'll utilize the `sentence-transformers` registry to load the model, unlocking its capabilities and preparing it for action:

```python
import torch
from lancedb.embeddings import get_registry

model_name = "plaguss/bge-base-argilla-sdk-matryoshka"
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model = get_registry().get("sentence-transformers").create(name=model_name, device=device)
```

### Create the table with the documentation chunks

With our fine-tuned model loaded, we're ready to create the table that will store our embeddings. To define the schema for this table, we'll employ a `LanceModel`, similar to `pydantic.BaseModel`, to create a robust representation of our `Docs` entity.

```python
from lancedb.pydantic import LanceModel, Vector

class Docs(LanceModel):
    query: str = model.SourceField()
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

table_name = "docs"
table = db.create_table(table_name, schema=Docs)
```

The previous code snippet sets the stage for creating a table with three essential columns:

- `query`: dedicated to storing the synthetic query

- `text`: housing the chunked documentation text

- `vector`: associated with the dimension from our fine-tuned model, ready to store the embeddings

With this table structure in place, we can now interact with the table.

#### Populate the table

With our table structure established, we're now ready to populate it with data. Let's load the final dataset, which contains the queries, and ingest them into our database, accompanied by their corresponding embeddings. This crucial step will bring our vector database to life, enabling our chatbot to store and retrieve relevant information efficiently:

```python
ds = load_dataset("plaguss/argilla_sdk_docs_queries", split="train")

batch_size = 50
for batch in tqdm.tqdm(ds.iter(batch_size), total=len(ds) // batch_size):
    embeddings = model.generate_embeddings(batch["positive"])
    df = pd.DataFrame.from_dict({"query": batch["positive"], "text": batch["anchor"], "vector": embeddings})
    table.add(df)
```

In the previous code snippet, we iterated over the dataset in batches, generating embeddings for the synthetic queries in the `positive` column using our fine-tuned model. We then created a Pandas dataframe, to include the `query`, `text`, and `vector` columns. This dataframe combines the `positive` and `anchor` columns with the freshly generated embeddings, respectively.

Now, let's put our vector database to the test! For a sample query, "How can I get the current user?" (using the Argilla SDK), we'll generate the embedding using our custom embedding model. We'll then search for the top 3 most similar occurrences in our table, leveraging the `cosine` metric to measure similarity. Finally, we'll extract the relevant `text` column, which corresponds to the chunk of documentation that best matches our query:

```python
query = "How can I get the current user?"
embedded_query = model.generate_embeddings([query])

retrieved = (
    table
        .search(embedded_query[0])
        .metric("cosine")
        .limit(3)
        .select(["text"])  # Just grab the chunk to use for context
        .to_list()
)
```

<details close>
<summary>Click to see the result</summary>
<br>

This would be the result:

```python
>>> retrieved
[{'text': 'python\nuser = client.users("my_username")\n\nThe current user of the rg.Argilla client can be accessed using the me attribute:\n\npython\nclient.me\n\nClass Reference\n\nrg.User\n\n::: argilla_sdk.users.User\n    options:\n        heading_level: 3',
  '_distance': 0.1881886124610901},
 {'text': 'python\nuser = client.users("my_username")\n\nThe current user of the rg.Argilla client can be accessed using the me attribute:\n\npython\nclient.me\n\nClass Reference\n\nrg.User\n\n::: argilla_sdk.users.User\n    options:\n        heading_level: 3',
  '_distance': 0.20238929986953735},
 {'text': 'Retrieve a user\n\nYou can retrieve an existing user from Argilla by accessing the users attribute on the Argilla class and passing the username as an argument.\n\n```python\nimport argilla_sdk as rg\n\nclient = rg.Argilla(api_url="", api_key="")\n\nretrieved_user = client.users("my_username")\n```',
  '_distance': 0.20401990413665771}]

>>> print(retrieved[0]["text"])
python
user = client.users("my_username")

The current user of the rg.Argilla client can be accessed using the me attribute:

python
client.me

Class Reference

rg.User

::: argilla_sdk.users.User
    options:
        heading_level: 3
```

</details>


Let's dive into the first row of our dataset and see what insights we can uncover. At first glance, it appears to contain information related to the query, which is exactly what we'd expect. To get the current user, we can utilize the `client.me` method. However, we also notice some extraneous content, which is likely a result of the chunking strategy employed. This strategy, while effective, could benefit from some refinement. By reviewing the dataset in Argilla, we can gain a deeper understanding of how to optimize our chunking approach, ultimately leading to a more streamlined dataset. For now, though, it seems like a solid starting point to build upon.

#### Store the database in the Hugging Face Hub

Now that we have a database, we will store it as another artifact in our dataset repository. You can visit the repo to find the functions that can help us, but it's as simple as running the following function:

```python
import Path
import os

local_dir = Path.home() / ".cache/argilla_sdk_docs_db"

upload_database(
    local_dir / "lancedb",
    repo_id="plaguss/argilla_sdk_docs_queries",
    token=os.getenv("HF_API_TOKEN")
)
```

The final step in our database storage journey is just a command away! By running the function, we'll create a brand new file called `lancedb.tar.gz`, which will neatly package our vector database. You can take a sneak peek at the resulting file in the [`plaguss/argilla_sdk_docs_queries`](https://huggingface.co/datasets/plaguss/argilla_sdk_docs_queries/tree/main) repository on the Hugging Face Hub, where it's stored alongside other essential files.

```python
db_path = download_database(repo_id)
```

The moment of truth has arrived! With our database successfully downloaded, we can now verify that everything is in order. By default, the file will be stored at `Path.home() / ".cache/argilla_sdk_docs_db"`, but can be easily customized. We can connect again to it and check everything works as expected:

```python
db = lancedb.connect(str(db_path))
table = db.open_table(table_name)

query = "how can I delete users?"

retrieved = (
    table
        .search(query)
        .metric("cosine")
        .limit(1)
        .to_pydantic(Docs)
)

for d in retrieved:
    print("======\nQUERY\n======")
    print(d.query)
    print("======\nDOC\n======")
    print(d.text)

# ======
# QUERY
# ======
# Is it possible to remove a user from Argilla by utilizing the delete function on the User class?
# ======
# DOC
# ======
# Delete a user

# You can delete an existing user from Argilla by calling the delete method on the User class.

# ```python
# import argilla_sdk as rg

# client = rg.Argilla(api_url="", api_key="")

# user_to_delete = client.users('my_username')

# deleted_user = user_to_delete.delete()
# ```
```

The database for the retrieval of documents is done, so let's go for the app!

## Creating our ChatBot

All the pieces are ready for our chatbot; we need to connect them and make them available in an interface.

### The Gradio App

Let's bring the RAG app to life! Using [gradio](https://www.gradio.app/), we can effortlessly create chatbot apps. In this case, we'll design a simple yet effective interface to showcase our chatbot's capabilities. To see the app in action, take a look at the [app.py](https://github.com/argilla-io/argilla-sdk-chatbot/blob/develop/app/app.py) script in the Argilla SDK Chatbot repository on GitHub.

Before we dive into the details of building our chatbot app, let's take a step back and admire the final result. With just a few lines of code, we've managed to create a user-friendly interface that brings our RAG chatbot to life.

![chatty](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-chatbot/img_1.png)

```python
import gradio as gr

gr.ChatInterface(
    chatty,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Ask me about the new argilla SDK", container=False, scale=7),
    title="Argilla SDK Chatbot",
    description="Ask a question about Argilla SDK",
    theme="soft",
    examples=[
        "How can I connect to an argilla server?",
        "How can I access a dataset?",
        "How can I get the current user?"
    ],
    cache_examples=True,
    retry_btn=None,
).launch()
```

And there you have it! If you're eager to learn more about creating your own chatbot, be sure to check out Gradio's excellent guide on [Chatbot with Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast). It's a treasure trove of knowledge that will have you building your own chatbot in no time.

Now, let's delve deeper into the inner workings of our `app.py` script. We'll break down the key components, focusing on the essential elements that bring our chatbot to life. To keep things concise, we'll gloss over some of the finer details.

First up, let's examine the `Database` class, the backbone of our chatbot's knowledge and functionality. This component plays a vital role in storing and retrieving the data that fuels our chatbot's conversations:

<details close>
<summary>Click to see Database class</summary>
<br>

```python
class Database:

    def __init__(self, settings: Settings) -> None:

        self.settings = settings
        self._table: lancedb.table.LanceTable = self.get_table_from_db()

    def get_table_from_db(self) -> lancedb.table.LanceTable:

        lancedb_db_path = self.settings.LOCAL_DIR / self.settings.LANCEDB

        if not lancedb_db_path.exists():
            lancedb_db_path = download_database(
                self.settings.REPO_ID,
                lancedb_file=self.settings.LANCEDB_FILE_TAR,
                local_dir=self.settings.LOCAL_DIR,
                token=self.settings.TOKEN,
            )

        db = lancedb.connect(str(lancedb_db_path))
        table = db.open_table(self.settings.TABLE_NAME)
        return table

    def retrieve_doc_chunks(
        self, query: str, limit: int = 12, hard_limit: int = 4
    ) -> str:

        # Embed the query to use our custom model instead of the default one.
        embedded_query = model.generate_embeddings([query])
        field_to_retrieve = "text"
        retrieved = (
            self._table.search(embedded_query[0])
            .metric("cosine")
            .limit(limit)
            .select([field_to_retrieve])  # Just grab the chunk to use for context
            .to_list()
        )
        return self._prepare_context(retrieved, hard_limit)

    @staticmethod
    def _prepare_context(retrieved: list[dict[str, str]], hard_limit: int) -> str:

        # We have repeated questions (up to 4) for a given chunk, so we may get repeated chunks.
        # Request more than necessary and filter them afterwards
        responses = []
        unique_responses = set()

        for item in retrieved:
            chunk = item["text"]
            if chunk not in unique_responses:
                unique_responses.add(chunk)
                responses.append(chunk)

        context = ""
        for i, item in enumerate(responses[:hard_limit]):
            if i > 0:
                context += "\n\n"
            context += f"---\n{item}"
        return context
```

</details><p>

With our `Database` class in place, we've successfully bridged the gap between our chatbot's conversational flow and the knowledge stored in our database. Now, let's bring everything together! Once we've downloaded our embedding model  (the script will do it automatically), we can instantiate the `Database` class, effectively deploying our database to the desired location - in this case, our Hugging Face Space.

This marks a major milestone in our chatbot development journey. With our database integrated and ready for action, we're just a step away from unleashing our chatbot's full potential.

```python
database = Database(settings=settings)  # The settings can be seen in the following snippet

context = database.retrieve_doc_chunks("How can I delete a user?", limit=2, hard_limit=1)

>>> print(context)
# ---
# Delete a user

# You can delete an existing user from Argilla by calling the delete method on the User class.

# ```python
# import argilla_sdk as rg

# client = rg.Argilla(api_url="", api_key="")

# user_to_delete = client.users('my_username')

# deleted_user = user_to_delete.delete()
# ```
```

<details close>
<summary>Click to see Settings class</summary>
<br>

```python
@dataclass
class Settings:
    LANCEDB: str = "lancedb"
    LANCEDB_FILE_TAR: str = "lancedb.tar.gz"
    TOKEN: str = os.getenv("HF_API_TOKEN")
    LOCAL_DIR: Path = Path.home() / ".cache/argilla_sdk_docs_db"
    REPO_ID: str = "plaguss/argilla_sdk_docs_queries"
    TABLE_NAME: str = "docs"
    MODEL_NAME: str = "plaguss/bge-base-argilla-sdk-matryoshka"
    DEVICE: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    MODEL_ID: str = "meta-llama/Meta-Llama-3-70B-Instruct"
```

</details>

The final piece of the puzzle is now in place - our database is ready to fuel our chatbot's conversations. Next, we need to prepare our model to handle the influx of user queries. This is where the power of [inference endpoints](https://huggingface.co/inference-endpoints/dedicated) comes into play. These dedicated endpoints provide a seamless way to deploy and manage our model, ensuring it's always ready to respond to user input.

Fortunately, working with inference endpoints is a breeze, thanks to the [`inference client`](https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi#inference-client) from the `huggingface_hub` library:

```python
def get_client_and_tokenizer(
    model_id: str = settings.MODEL_ID, tokenizer_id: Optional[str] = None
) -> tuple[InferenceClient, AutoTokenizer]:
    if tokenizer_id is None:
        tokenizer_id = model_id

    client = InferenceClient()
    base_url = client._resolve_url(model=model_id, task="text-generation")
    # Note: We could move to the AsyncClient
    client = InferenceClient(model=base_url, token=os.getenv("HF_API_TOKEN"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    return client, tokenizer

# Load the client and tokenizer
client, tokenizer = get_client_and_tokenizer()
```

With our components in place, we've reached the stage of preparing the prompt that will be fed into our client. This prompt will serve as the input that sparks the magic of our machine learning model, guiding it to generate a response that's both accurate and informative, while avoiding answering unrelated questions. In this section, we'll delve into the details of crafting a well-structured prompt that sets our model up for success. The `prepare_input` function will prepare the conversation, applying the prompt and the chat template to be passed to the model:

```python
def prepare_input(message: str, history: list[tuple[str, str]]) -> str:

    # Retrieve the context from the database
    context = database.retrieve_doc_chunks(message)

    # Prepare the conversation for the model.
    conversation = []
    for human, bot in history:
        conversation.append({"role": "user", "content": human})
        conversation.append({"role": "assistant", "content": bot})

    conversation.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    conversation.append(
        {
            "role": "user",
            "content": ARGILLA_BOT_TEMPLATE.format(message=message, context=context),
        }
    )

    return tokenizer.apply_chat_template(
        [conversation],
        tokenize=False,
        add_generation_prompt=True,
    )[0]
```

This function will take two arguments: `message` and `history` courtesy of the gradio [`ChatInterface`](https://www.gradio.app/docs/gradio/chatinterface), obtain the documentation pieces from the database to help the LLM with the response, and prepare the prompt to be passed to our `LLM` model.

<details close>
<summary>Click to see the system prompt and the bot template</summary>
<br>

These are the `system_prompt` and the prompt template used. They are heavily inspired by [`wandbot`](https://github.com/wandb/wandbot) from Weights and Biases.

````python
SYSTEM_PROMPT = """\
You are a support expert in Argilla SDK, whose goal is help users with their questions.
As a trustworthy expert, you must provide truthful answers to questions using only the provided documentation snippets, not prior knowledge.
Here are guidelines you must follow when responding to user questions:

##Purpose and Functionality**
- Answer questions related to the Argilla SDK.
- Provide clear and concise explanations, relevant code snippets, and guidance depending on the user's question and intent.
- Ensure users succeed in effectively understanding and using Argilla's features.
- Provide accurate responses to the user's questions.

**Specificity**
- Be specific and provide details only when required.
- Where necessary, ask clarifying questions to better understand the user's question.
- Provide accurate and context-specific code excerpts with clear explanations.
- Ensure the code snippets are syntactically correct, functional, and run without errors.
- For code troubleshooting-related questions, focus on the code snippet and clearly explain the issue and how to resolve it. 
- Avoid boilerplate code such as imports, installs, etc.

**Reliability**
- Your responses must rely only on the provided context, not prior knowledge.
- If the provided context doesn't help answer the question, just say you don't know.
- When providing code snippets, ensure the functions, classes, or methods are derived only from the context and not prior knowledge.
- Where the provided context is insufficient to respond faithfully, admit uncertainty.
- Remind the user of your specialization in Argilla SDK support when a question is outside your domain of expertise.
- Redirect the user to the appropriate support channels - Argilla [community](https://join.slack.com/t/rubrixworkspace/shared_invite/zt-whigkyjn-a3IUJLD7gDbTZ0rKlvcJ5g) when the question is outside your capabilities or you do not have enough context to answer the question.

**Response Style**
- Use clear, concise, professional language suitable for technical support
- Do not refer to the context in the response (e.g., "As mentioned in the context...") instead, provide the information directly in the response.

**Example**:

The correct answer to the user's query

 Steps to solve the problem:
 - **Step 1**: ...
 - **Step 2**: ...
 ...

 Here's a code snippet

 ```python
 # Code example
 ...
 ```
 
 **Explanation**:

 - Point 1
 - Point 2
 ...
"""

ARGILLA_BOT_TEMPLATE = """\
Please provide an answer to the following question related to Argilla's new SDK.

You can make use of the chunks of documents in the context to help you generating the response.

## Query:
{message}

## Context:
{context}
"""
````

</details>

We've reached the culmination of our conversational AI system: the `chatty` function. This function serves as the orchestrator, bringing together the various components we've built so far. Its primary responsibility is to invoke the `prepare_input` function, which crafts the prompt that will be passed to the client. Then, we yield the stream of text as it's being generated, and once the response is finished, the conversation history will be saved, providing us with a valuable resource to review and refine our model, ensuring it continues to improve with each iteration.

```python
def chatty(message: str, history: list[tuple[str, str]]) -> Generator[str, None, None]:
    prompt = prepare_input(message, history)

    partial_response = ""

    for token_stream in client.text_generation(prompt=prompt, **client_kwargs):
        partial_response += token_stream
        yield partial_response

    global conv_id
    new_conversation = len(history) == 0
    if new_conversation:
        conv_id = str(uuid.uuid4())
    else:
        history.append((message, None))

    # Register to argilla dataset
    argilla_dataset.records.log(
        [
            {
                "instruction": create_chat_html(history) if history else message,
                "response": partial_response,
                "conv_id": conv_id,
                "turn": len(history)
            },
        ]
    )
```

The moment of truth has arrived! Our app is now ready to be put to the test. To see it in action, simply run `python app.py` in your local environment. But before you do, make sure you have access to a deployed model at an inference endpoint. In this example, we're using the powerful Llama 3 70B model, but feel free to experiment with other models that suit your needs. By tweaking the model and fine-tuning the app, you can unlock its full potential and explore new possibilities in AI development.

### Deploy the ChatBot app on Hugging Face Spaces

---
Now that our app is up and running, it's time to share it with the world! To deploy our app and make it accessible to others, we'll follow the steps outlined in [Gradio's guide](https://www.gradio.app/guides/sharing-your-app) to sharing your app. Our chosen platform for hosting is Hugging Face Spaces, a fantastic tool for showcasing AI-powered projects.

To get started, we'll need to add a `requirements.txt` file to our repository, which lists the dependencies required to run our app. This is a crucial step in ensuring that our app can be easily reproduced and deployed. You can learn more about managing dependencies in Hugging Face Spaces [spaces dependencies](https://huggingface.co/docs/hub/spaces-dependencies).

Next, we'll need to add our Hugging Face API token as a secret, following the instructions in [this guide](https://huggingface.co/docs/hub/spaces-overview#managing-secrets). This will allow our app to authenticate with the Hugging Face ecosystem.

Once we've uploaded our `app.py` file, our Space will be built, and we'll be able to access our app at the following link:

> https://huggingface.co/spaces/plaguss/argilla-sdk-chatbot-space

Take a look at our example Space files here to see how it all comes together. By following these steps, you'll be able to share your own AI-powered app with the world and collaborate with others in the Hugging Face community.

### Playing around with our ChatBot

We can now put the Chatbot to the test. We've provided some default queries to get you started, but feel free to experiment with your own questions. For instance, you could ask: `What are the Settings in the new SDK?`

As you can see from the screenshot below, our chatbot is ready to provide helpful responses to your queries:

![chatbot img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-chatbot/chatbot.png)

But that's not all! You can also challenge our chatbot to generate settings for a specific dataset, like the one we created earlier in this tutorial. For example, you could ask it to suggest settings for a dataset designed to fine-tune an embedding model, similar to the one we explored in the [An Argilla dataset with triplets to fine-tune an embedding model](#an-A:rgilla-dataset-with-triplets-to-fine-tune-an-embedding-model) section.

Take a look at the screenshot below to see how our chatbot responds to this type of query.

![chatbot sentence-embedding](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/argilla-chatbot/chatbot-sentence-embeddings.png)

Go ahead, ask your questions, and see what insights our chatbot can provide!

## Next steps

In this tutorial, we've successfully built a chatbot that can provide helpful responses to questions about the Argilla SDK and its applications. By leveraging the power of Llama 3 70B and Gradio, we've created a user-friendly interface that can assist developers in understanding how to work with datasets and fine-tune embedding models.

However, our chatbot is just the starting point, and there are many ways we can improve and expand its capabilities. Here are some possible next steps to tackle:

- Improve the chunking strategy: Experiment with different chunking strategies, parameters, and sizes to optimize the chatbot's performance and response quality.

- Implement deduplication and filtering: Add deduplication and filtering mechanisms to the training dataset to remove duplicates and irrelevant information, ensuring that the chatbot provides accurate and concise responses.

- Include sources for responses: Enhance the chatbot's responses by including links to relevant documentation and sources, allowing users to dive deeper into the topics and explore further.

By addressing these areas, we can take our chatbot to the next level, making it an even more valuable resource for developers working with the Argilla SDK. The possibilities are endless, and we're excited to see where this project will go from here. Stay tuned for future updates and improvements!
