---
title: "Text-Generation Pipeline on Habana Gaudi2 Accelerator" 
thumbnail: /blog/assets/textgen-pipe-gaudi/thumbnail.png
authors:
- user: siddjags
  guest: true
---

# Text-Generation Pipeline on Habana Gaudi2 Accelerator
With the Generative AI (GenAI) revolution in full swing, text-generation with open-source transformer models like Llama 2 has become the talk of the town. AI enthusiasts as well as developers are looking to leverage the generative abilities of such models for their own use cases and applications. This article shows how easy it is to generate text with the Llama 2 family of models (7b, 13b and 70b) using Optimum Habana and a custom pipeline class – you'll be able to run the models with just a few lines of code!

## Prerequisites
Since the Llama 2 models are part of a gated repo, you need to request access if you haven't done it already. First, you have to visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and accept the terms and conditions. After you are granted access by Meta (it can take a day or two), you have to request access [in Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf), using the same email address you provided in the Meta form.

After you are granted access, please login to your Hugging Face account by running the following command (you will need an access token, which you can get from [your user profile page](https://huggingface.co/settings/tokens)):
Since the Llama-2 models are part of a gated repo, you need to request access [here](https://huggingface.co/meta-llama/Llama-2-7b-hf). Make sure that the email address you provide is the same as your Hugging Face account. After you are granted access, login to your Hugging Face account by running the following command (you will require an access token).

```bash
huggingface-cli login
```

You also need to install the latest version of Optimum Habana and clone the repo to access the pipeline script. Here are the commands to do so:

```bash
pip install --upgrade-strategy eager optimum[habana]
```

In case you are planning to run distributed inference, install DeepSpeed depending on your SynapseAI version. In this case, I am using SynapseAI 1.14.0.

```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
```

Now you are all set to perform text-generation with the pipeline!

## Using the Pipeline
First, go to the following directory in your `optimum-habana` checkout where the pipeline scripts are located, and follow the instructions in the `README` to update your `PYTHONPATH`.

```bash
cd optimum-habana/examples/text-generation/text-generation-pipeline
```

If you wish to generate a sequence of text from a prompt of your choice, here is a sample command.

```bash
python run_pipeline.py  --model_name_or_path meta-llama/Llama-2-7b-hf --use_hpu_graphs --use_kv_cache --max_new_tokens 100 --do_sample --prompt "Here is my prompt"
```

You can also pass multiple prompts as input and change the temperature and top_p values for generation as follows.

```bash
python run_pipeline.py --model_name_or_path meta-llama/Llama-2-13b-hf --use_hpu_graphs --use_kv_cache --max_new_tokens 100 --do_sample --temperature 0.5 --top_p 0.95 --prompt "Hello world" "How are you?"
```

For generating text with large models such as Llama-2-70b, here is a sample command to launch the pipeline with DeepSpeed.

```bash
python ../../gaudi_spawn.py --use_deepspeed --world_size 8 run_pipeline.py --model_name_or_path meta-llama/Llama-2-70b-hf --max_new_tokens 100 --bf16 --use_hpu_graphs --use_kv_cache --do_sample --temperature 0.5 --top_p 0.95 --prompt "Hello world" "How are you?" "Here is my prompt" "Once upon a time"
```

## Usage in Python Scripts

You can use the pipeline class in your own scripts as shown in the example below. Run the following sample script from `optimum-habana/examples/text-generation/text-generation-pipeline`.
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

> You will have to run the above script with `python <name_of_script>.py --model_name_or_path a_model_name` as `--model_name_or_path` is a required argument. However, the model name can be programatically changed as shown in the python snippet.

This shows us that the pipeline class operates on a string input and performs data pre-processing as well as post-processing for us.

## LangChain Compatibility

The text-generation pipeline can be fed as input to LangChain classes via the `use_with_langchain` constructor argument. You can install LangChain as follows.
```bash
pip install langchain==0.0.191
```

Here is a sample script that shows how the pipeline class can be used with LangChain.
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
> The pipeline class has been validated for LangChain version 0.0.191 and may not work with other versions of the package.

## Conclusion

We presented a custom text-generation pipeline on Habana Gaudi2 that accepts single or multiple prompts as input. This pipeline offers great flexibility in terms of model size as well as parameters affecting text-generation quality. Furthermore, it is also very easy to use and to plug into your scripts, and is compatible with LangChain.
