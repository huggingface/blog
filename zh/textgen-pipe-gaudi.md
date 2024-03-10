---
title: "基于英特尔® Gaudi® 2 AI 加速器的文本生成流水线" 
thumbnail: /blog/assets/textgen-pipe-gaudi/thumbnail.png
authors:
- user: siddjags
  guest: true
translators:
- user: MatrixYao
---

# 基于英特尔® Gaudi® 2 AI 加速器的文本生成流水线

随着生成式人工智能（Generative AI，GenAI）革命的全面推进，使用 Llama 2 等开源 transformer 模型生成文本已成为新风尚。人工智能爱好者及开发人员正在寻求利用此类模型的生成能力来赋能不同的场景及应用。本文展示了如何基于 Optimum Habana 以及我们实现的流水线类轻松使用 Llama 2 系列模型（7b、13b 及 70b）生成文本 - 仅需几行代码，即可运行！

我们设计并实现了一个旨在为用户提供极大的灵活性和易用性流水线类。它提供了高层级的抽象以支持包含预处理和后处理在内的端到端文本生成。同时，用户也可以通过多种方法使用该流水线类 - 你可以在 Optimum Habana 代码库中直接运行 `run_pipeline.py` 脚本，也可以在你自己的 python 脚本中调用该流水线类，还可以用该流水线类来初始化 LangChain。

## 准备工作

由于 Llama 2 模型实行的是许可式访问，因此如果你尚未申请访问权限，需要首先申请访问权限。方法如下：首先，访问 [Meta 网站](https://ai.meta.com/resources/models-and-libraries/llama-downloads)并接受相应条款。一旦 Meta 授予你访问权限（可能需要一两天），你需要使用你当时使用的电子邮箱地址申请 [Hugging Face Llama 2 模型库](https://huggingface.co/meta-llama/Llama-2-7b-hf)的访问权限。

获取访问权限后，可通过运行以下命令登录你的 Hugging Face 帐户（此时会需要一个访问令牌，你可从[你的用户个人资料页面](https://huggingface.co/settings/tokens)上获取）：

```bash
huggingface-cli login
```

你还需要安装最新版本的 Optimum Habana 并拉取其代码库以获取后续要使用的脚本。命令如下：

```bash
pip install optimum-habana==1.10.4
git clone -b v1.10-release https://github.com/huggingface/optimum-habana.git
```

如果想运行分布式推理，还需要根据你的 SynapseAI 版本安装对应的 DeepSpeed。在本例中，我使用的是 SynapseAI 1.14.0。

```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
```

至此，准备完毕！

## 方法一：通过命令直接使用流水线脚本

首先，使用如下命令进入 `optimum-habana` 的相应目录，然后按照 `README` 中的说明更新 `PYTHONPATH`。

```bash
cd optimum-habana/examples/text-generation
pip install -r requirements.txt
cd text-generation-pipeline
```

如果你想用自己的提示生成文本序列，下面给出了一个示例：

```bash
python run_pipeline.py  --model_name_or_path meta-llama/Llama-2-7b-hf --use_hpu_graphs --use_kv_cache --max_new_tokens 100 --do_sample --prompt "Here is my prompt"
```

你还可以传入多个提示作为输入，并更改生成的温度或 `top_p` 值，如下所示：

```bash
python run_pipeline.py --model_name_or_path meta-llama/Llama-2-13b-hf --use_hpu_graphs --use_kv_cache --max_new_tokens 100 --do_sample --temperature 0.5 --top_p 0.95 --prompt "Hello world" "How are you?"
```

如果想用 Llama-2-70b 等大尺寸模型生成文本，下面给出了一个用 DeepSpeed 启动流水线的示例命令：

```bash
python ../../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py --model_name_or_path meta-llama/Llama-2-70b-hf --max_new_tokens 100 --bf16 --use_hpu_graphs --use_kv_cache --do_sample --temperature 0.5 --top_p 0.95 --prompt "Hello world" "How are you?" "Here is my prompt" "Once upon a time"
```

## 方法二：在自己的 Python 脚本中调用流水线类

你还可以在自己的 Python 脚本中调用我们实现的流水线类，如下例所示。你需要在 `optimum-habana/examples/text-generation/text- generation-pipeline` 目录下运行该示例脚本[译者注：原因是 `GaudiTextGenerationPipeline` 这个类的定义在该目录的 `pipeline.py` 中]。

```python
import argparse
import logging

from pipeline import GaudiTextGenerationPipeline
from run_generation import setup_parser

# Define a logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set up an argument parser
parser = argparse.ArgumentParser()
args = setup_parser(parser)

# Define some pipeline arguments. Note that --model_name_or_path is a required argument for this script
args.num_return_sequences = 1
args.model_name_or_path = "meta-llama/Llama-2-7b-hf"
args.max_new_tokens = 100
args.use_hpu_graphs = True
args.use_kv_cache = True
args.do_sample = True

# Initialize the pipeline
pipe = GaudiTextGenerationPipeline(args, logger)

# You can provide input prompts as strings
prompts = ["He is working on", "Once upon a time", "Far far away"]

# Generate text with pipeline
for prompt in prompts:
    print(f"Prompt: {prompt}")
    output = pipe(prompt)
    print(f"Generated Text: {repr(output)}")
```

> 你需要用 `python <name_of_script>.py --model_name_or_path a_model_name` 命令来运行上述脚本，其中 `--model_name_or_path` 是必需的参数。当然，你也可以在代码中直接更改模型名称（如上述 Python 代码片段所示）。

上述代码段表明我们实现的流水线类 `GaudiTextGenerationPipeline` 会对输入字符串执行生成文本所需的全部操作，包括数据预处理及后处理在内。

## 方法二：在 LangChain 中使用流水线类

如果在构造时传入 `use_with_langchain` 参数的话，我们的文本生成流水线还可以作为 LangChain 的兼容组件使用。首先，按照如下方式安装 LangChain：

```bash
pip install langchain==0.0.191
```

下面给出了一个如何在 LangChain 中使用我们的流水线类的代码示例。

```python
import argparse
import logging

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from pipeline import GaudiTextGenerationPipeline
from run_generation import setup_parser

# Define a logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set up an argument parser
parser = argparse.ArgumentParser()
args = setup_parser(parser)

# Define some pipeline arguments. Note that --model_name_or_path is a required argument for this script
args.num_return_sequences = 1
args.model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
args.max_input_tokens = 2048
args.max_new_tokens = 1000
args.use_hpu_graphs = True
args.use_kv_cache = True
args.do_sample = True
args.temperature = 0.2
args.top_p = 0.95

# Initialize the pipeline
pipe = GaudiTextGenerationPipeline(args, logger, use_with_langchain=True)

# Create LangChain object
llm = HuggingFacePipeline(pipeline=pipe)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
just say that you don't know, don't try to make up an answer.

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: {question}
Answer: """

prompt = PromptTemplate(input_variables=["question"], template=template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Use LangChain object
question = "Which libraries and model providers offer LLMs?"
response = llm_chain(prompt.format(question=question))
print(f"Question 1: {question}")
print(f"Response 1: {response['text']}")

question = "What is the provided context about?"
response = llm_chain(prompt.format(question=question))
print(f"\nQuestion 2: {question}")
print(f"Response 2: {response['text']}")
```

> 该流水线类当前仅在 LangChain 0.0.191 版上验证通过，其他版本可能不兼容。

## 总结

我们在英特尔® Gaudi® 2 AI 加速器上实现了一个自定义的文本生成流水线，其可接受单个或多个提示作为输入。该流水线类灵活支持各种模型尺寸及各种影响文本生成质量参数。此外，不管是直接使用还是将它插入你自己的脚本都非常简单，并且其还与 LangChain 兼容。

> 使用预训练模型需遵守第三方许可，如 “Llama 2 社区许可协议”(LLAMAV2)。有关 LLAMA2 模型的预期用途有哪些、哪些行为会被视为滥用或超范围使用、预期使用者是谁以及其他条款，请仔细阅读此[链接](https://ai.meta.com/llama/license/)中的说明。用户需自主承担遵守任何第三方许可的责任和义务，Habana Labs 不承担任何与用户使用或遵守第三方许可相关的责任。

为了能够运行像 `Llama-2-70b-hf` 这样的受限模型，你需要：
> * 有一个 Hugging Face 帐户
> * 同意 HF Hub 上模型卡中的模型使用条款
> * 设好访问令牌
> * 使用 HF CLI 登录你的帐户，即在启动脚本之前运行 `huggingface-cli login`

> 英文原文: <url> https://huggingface.co/blog/textgen-pipe-gaudi </url>
> 原文作者：Siddhant Jagtap
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。
