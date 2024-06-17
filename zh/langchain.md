---
title: "Hugging Face x LangChain：全新 LangChain 合作伙伴包" 
thumbnail: /blog/assets/langchain_huggingface/thumbnail.png
authors:
- user: jofthomas
- user: kkondratenko
  guest: true
- user: efriis
  guest: true
  org: langchain-ai
translators:
- user: MatrixYao
- user: zhongdongy
  proofreader: true
---

# Hugging Face x LangChain: 全新 LangChain 合作伙伴包

我们很高兴官宣发布 **`langchain_huggingface`**，这是一个由 Hugging Face 和 LangChain 共同维护的 LangChain 合作伙伴包。这个新的 Python 包旨在将 Hugging Face 最新功能引入 LangChain 并保持同步。

# 源自社区，服务社区

目前，LangChain 中所有与 Hugging Face 相关的类都是由社区贡献的。虽然我们以此为基础蓬勃发展，但随着时间的推移，其中一些类在设计时由于缺乏来自 Hugging Face 的内部视角而在后期被废弃。

通过 Langchain 合作伙伴包这个方式，我们的目标是缩短将 Hugging Face 生态系统中的新功能带给 LangChain 用户所需的时间。

**`langchain-huggingface`** 与 LangChain 无缝集成，为在 LangChain 生态系统中使用 Hugging Face 模型提供了一种可用且高效的方法。这种伙伴关系不仅仅涉及到技术贡献，还展示了双方对维护和不断改进这一集成的共同承诺。

## **起步**

**`langchain-huggingface`** 的起步非常简单。以下是安装该 [软件包](https://github.com/langchain-ai/langchain/tree/master/libs/partners/huggingface) 的方法:

```python
pip install langchain-huggingface
```

现在，包已经安装完毕，我们来看看里面有什么吧！

## LLM 文本生成

### HuggingFacePipeline

`transformers` 中的 [Pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) 类是 Hugging Face 工具箱中最通用的工具。LangChain 的设计主要是面向 RAG 和 Agent 应用场景，因此，在 Langchain 中流水线被简化为下面几个以文本为中心的任务: `文本生成` 、 `文生文` 、 `摘要` 、 `翻译` 等。

用户可以使用 `from_model_id` 方法直接加载模型:

```python
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
)
llm.invoke("Hugging Face is")
```

也可以自定义流水线，再传给 `HuggingFacePipeline` 类:

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

使用 `HuggingFacePipeline` 时，模型是加载至本机并在本机运行的，因此你可能会受到本机可用资源的限制。

### HuggingFaceEndpoint

该类也有两种方法。你可以使用 `repo_id` 参数指定模型。也可以使用 `endpoint_url` 指定服务终端，这些终端使用 [无服务器 API](https://huggingface.co/inference-api/serverless)，这对于有 Hugging Face [专业帐户](https://huggingface.co/subscribe/pro) 或 [企业 hub](https://huggingface.co/enterprise) 的用户大有好处。普通用户也可以通过在代码环境中设置自己的 HF 令牌从而在免费请求数配额内使用终端。

```python
from langchain_huggingface import HuggingFaceEndpoint

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

该类在底层实现时使用了 [InferenceClient](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client)，因此能够为已部署的 TGI 实例提供面向各种用例的无服务器 API。

### ChatHuggingFace

每个模型都有最适合自己的特殊词元。如果没有将这些词元添加到提示中，将大大降低模型的表现。

为了把用户的消息转成 LLM 所需的提示，大多数 LLM 分词器中都提供了一个名为 [chat_template](https://huggingface.co/docs/transformers/chat_templated) 的成员属性。

要了解不同模型的 `chat_template` 的详细信息，可访问我创建的 [space](https://huggingface.co/spaces/Jofthomas/Chat_template_viewer)！

`ChatHuggingFace` 类对 LLM 进行了包装，其接受用户消息作为输入，然后用 `tokenizer.apply_chat_template` 方法构造出正确的提示。

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    endpoint_url="<endpoint_url>",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=False,
)
llm_engine_hf = ChatHuggingFace(llm=llm)
llm_engine_hf.invoke("Hugging Face is")
```

上述代码等效于:

```python
# with mistralai/Mistral-7B-Instruct-v0.2
llm.invoke("<s>[INST] Hugging Face is [/INST]")

# with meta-llama/Meta-Llama-3-8B-Instruct
llm.invoke("""<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hugging Face is<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
```

## 嵌入

Hugging Face 里有很多非常强大的嵌入模型，你可直接把它们用于自己的流水线。

首先，选择你想要的模型。关于如何选择嵌入模型，一个很好的参考是 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard)。

### HuggingFaceEmbeddings

该类使用 [sentence-transformers](https://sbert.net/) 来计算嵌入。其计算是在本机进行的，因此需要使用你自己的本机资源。

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

`HuggingFaceEndpointEmbeddings` 与 `HuggingFaceEndpoint` 对 LLM 所做的非常相似，其在实现上也是使用 InferenceClient 来计算嵌入。它可以与 hub 上的模型以及 TEI 实例一起使用，TEI 实例无论是本地部署还是在线部署都可以。

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

## 总结

我们致力于让 **`langchain-huggingface`** 变得越来越好。我们将积极监控反馈和问题，并努力尽快解决它们。我们还将不断添加新的特性和功能，以拓展该软件包使其支持更广泛的社区应用。我们强烈推荐你尝试 `langchain-huggingface` 软件包并提出宝贵意见，有了你的支持，这个软件包的未来道路才会越走越宽。
