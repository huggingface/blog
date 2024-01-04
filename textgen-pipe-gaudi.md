---
title: "Text-Generation Pipeline on Habana Gaudi2 Accelerator" 
thumbnail: /blog/assets/textgen-pipe-gaudi/thumbnail.gif
authors:
- user: siddjags
---

# Text-Generation Pipeline on Habana Gaudi2 Accelerator

With the GenAI revolution in full swing, text-generation with open-source transformer models like Llama-2 has become the talk of the town. AI enthusiasts as well as developers are looking to leverage the generative abilities of such models for their own use cases and applications. This article will demonstrate how easy it is to generate text with the Llama-2 family of models (7b, 13b and 70b) using Optimum Habana and a custom pipeline class. You will then be able to generate text with only a few lines of code.

## Prerequisites

Since the Llama-2 models are part of a gated repo, you need to request access [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). Make sure that the email address you provide is the same as your Hugging Face account. After you are granted access, login to your Hugging Face account by running the following command (You will require an access token).

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

## Conclusion

In this blog, we presented a custom text-generation pipeline that accepts single as well as multiple prompts as input. This pipeline offers great flexibility in terms of model size as well as parameters affecting text-generation quality.