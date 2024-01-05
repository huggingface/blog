---
title: "Text-Generation Pipeline on Habana Gaudi2 Accelerator" 
thumbnail: /blog/assets/textgen-pipe-gaudi/thumbnail.png
authors:
- user: siddjags
---

# Text-Generation Pipeline on Habana Gaudi2 Accelerator

With the Generative AI (GenAI) revolution in full swing, text-generation with open-source transformer models like Llama-2 has become the talk of the town. AI enthusiasts as well as developers are looking to leverage the generative abilities of such models for their own use cases and applications. This article will demonstrate how easy it is to generate text with the Llama-2 family of models (7b, 13b and 70b) using Optimum Habana and a custom pipeline class. You will then be able to generate text with only a few lines of code.

## Prerequisites

Since the Llama-2 models are part of a gated repo, you need to request access [here](https://huggingface.co/meta-llama/Llama-2-7b-hf). Make sure that the email address you provide is the same as your Hugging Face account. After you are granted access, login to your Hugging Face account by running the following command (you will require an access token).

```bash
huggingface-cli login
```

You also need to install the latest version of Optimum Habana and clone the repo to access the pipeline script. Here are the commands to do so:

```bash
pip install --upgrade-strategy eager optimum[habana]
git clone https://github.com/huggingface/optimum-habana.git
```

In case you are planning to run distributed inference, install DeepSpeed depending on SynapseAI version. In this case, I am using SynapseAI 1.13.0.

```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.13.0
```

Now you are all set to perform text-generation with the pipeline!

## Using the Pipeline
Run the following command to access the pipeline scripts and follow the instructions provided in the README to update your `PYTHONPATH`.

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

Last but not the least, you can use the pipeline class in your own scripts as shown in the example below. Run the following python snippet from `optimum-habana/examples/text-generation/text-generation-pipeline`.
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

Note: You will have to run the above script with `python <name_of_script>.py --model_name_or_path gpt2` as `--model_name_or_path` is a required argument. However, the model name can be programatically changed as shown in the python snippet.

This shows us that the pipeline class operates on a string input and performs data pre-processing as well as post-processing for us.

## Conclusion

In this blog, we presented a custom text-generation pipeline that accepts single as well as multiple prompts as input. This pipeline offers great flexibility in terms of model size as well as parameters affecting text-generation quality.