---
title: "Hugging Face x LangChain : A new partner package in Langchain" 
thumbnail: /blog/assets/langchain_huggingface/thumbnail.png
authors:
- user: jofthomas
- user: kkondratenko
  guest: true
- user: efriis
  guest: true
  org: langchain-ai
---
# Hugging Face x LangChain : A new partner package in LangChain

We are thrilled to announce the launch of **`langchain_huggingface`**, a partner package in LangChain jointly maintained by Hugging Face and LangChain. This new Python package is designed to bring the power of the latest development of Hugging Face into LangChain and keep it up to date. 

# From the community, for the community

All Huggingface related classes in LangChain were coded by the community, and while this is something we thrived for, over time some of them became deprecated because of the lack of insider’s perspective.

By becoming a partner package, we aim at reducing the time to bring new features available in the Hugging Face ecosystem to LangChain’s users.

**`langchain_huggingface`** integrates seamlessly with LangChain, providing an efficient and effective way to utilize Hugging Face models within the LangChain ecosystem. This partnership is not just about sharing technology but also about a joint commitment to maintain and continually improve this integration.

## **Getting Started**

Getting started with **`langchain_huggingface`** is straightforward. Here’s how you can install and begin using the package:

```python
!pip install langchain_huggingface
```

Now that the package is installed, let’s have a tour of what’s inside !

## The LLMs

### HuggingFacePipeline

Being in `transformers` the [Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines), is most versatile tool in the Hugging Face toolbox. LangChain being designed primarly to adress RAG and Agents use-cases, the scope of the pipeline here is reduced to the following text centric tasks `“text-generation"` , `"text2text-generation"`, `"summarization"`, `"translation”`.

Models can be loaded directly with the `from_model_id` method:

```python
from langchain_huggingface.llms import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
        },
)
llm.invoke("Hugging Face is")
```

Or you can also define the pipeline yourself before passing it to the class:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline

model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    #attn_implementation="flash_attention_2", # if you have an ampere GPU
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, top_k=50, temperature=0.1)
llm = HuggingFacePipeline(pipeline=pipe)
llm.invoke("Hugging Face is")
```

When using this class, the model will be loaded in cache and use your computer’s hardware, hence you may be limited by the available ressources on your computer.

### HuggingFaceEndpoint

There is also two ways to use this class. Either by specifying the model with the `repo_id` parameter. Those endpoints use the [serverless API](https://huggingface.co/inference-api/serverless) which is particularly beneficial to people using [pro accounts](https://huggingface.co/subscribe/pro) or [enterprise hub](https://huggingface.co/enterprise) but regular users can already have access to a fair amount of request by connecting with their HF token in the environment where they are executing the code.

```python
from langchain_huggingface.llms import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)
llm.invoke("Hugging Face is")
```

```python
llm = HuggingFaceEndpoint(
    endpoint_url="<endpoint_url>",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
)
llm.invoke("Hugging Face is")
```

Under the hood, this class uses the [InferenceClient](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client) to be able to serve a wide variety of use-case, from serverless api, to locally deployed TGI instances.

### ChatHuggingFace

Every model has its own special tokens with which it works best. And by not adding those special tokens to your prompt, your model will greatly underperform !

When going from a list of messages to a completion prompt, there is an attribute that exists in most LLM tokenizers called [chat_template](https://huggingface.co/docs/transformers/chat_templating).

To learn more about chat_template in the different models, visit this [space](https://huggingface.co/spaces/Jofthomas/Chat_template_viewer) I made!

This class is wrapper around the other LLMs. It takes as input a list of messages an then creates the correct completion prompt by using the `tokenizer.apply_chat_template` method.

```python
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    endpoint_url="<endpoint_url>",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
)
llm_engine_hf = ChatHuggingFace(llm=llm)
llm_engine_hf.invoke("Hugging Face is")
```

The code above is equivalent to :

```python
# with mistralai/Mistral-7B-Instruct-v0.2
llm.invoke("<s>[INST] Hugging Face is [/INST]")

# with meta-llama/Meta-Llama-3-8B-Instruct
llm.invoke("""<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hugging Face is<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
```

## The Embeddings

Hugging Face is filled with very powerful embedding models than you can directly leverage in your pipeline.

First choose your model, one good ressource for that is the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

### HuggingFaceEmbeddings

This class uses [sentence-transformers](https://sbert.net/) embeddings. it computes the embedding locally, hence using your computer ressources.

```python
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

model_name = "mixedbread-ai/mxbai-embed-large-v1"
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
)
texts = ["Hello, world!", "How are you?"]
hf_embeddings.embed_documents(texts)
```

### HuggingFaceEndpointEmbeddings

`HuggingFaceEndpointEmbeddings` is very similar to what `HuggingFaceEndpoint`  does for LLM,  in the sense that it also uses the InferenceClient under the hood to compute the embeddings.
It can be used with models on the hub, and TEI instances weither they are deployed locally or online. 

```python
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model= "mixedbread-ai/mxbai-embed-large-v1",
    task="feature-extraction",
    huggingfacehub_api_token="<HF_TOKEN>",
)
texts = ["Hello, world!", "How are you?"]
hf_embeddings.embed_documents(texts)
```

## Conclusion

We are committed to making **`langchain_huggingface`** better by the day. We will be actively monitoring feedback and issues, and working to address them as quickly as possible. We will also be adding new features and functionality, and expanding the package to support an even wider range of use cases.
