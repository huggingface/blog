---
title: "CodeGemma - an official Google release for code LLMs" 
thumbnail: /blog/assets/codegemma/thumbnail.jpg
authors:
- user: pcuenq
- user: osanseviero
- user: reach-vb
- user: philschmid
- user: mishig
- user: loubnabnl
---

# CodeGemma - an official Google release for code LLMs

CodeGemma is a family of open-access versions of Gemma specialized in code, and we’re excited to collaborate with Google on its release to make it as accessible as possible.🤗

CodeGemma comes in three flavors:

- A 2B base model specialized in infilling and open-ended generation.
- A 7B base model trained with both code infilling and natural language.
- A 7B instruct model a user can chat with about code.

We’ve collaborated with Google to ensure the best integration into the Hugging Face ecosystem. You can find the three open-access models ready to use on the Hub. Among the features and integrations being released, we have:

- [Models on the Hub](https://huggingface.co/models?search=google/codegemma), with their model cards and licenses. There are versions for the transformers library, checkpoints for use with Google’s original codebases, and full-precision GGUF files that the community can quantize.
- Transformers integration
- Integration with Google Cloud
- Integration with Inference Endpoints
- Code benchmarks

## Table of contents

  - [What is CodeGemma](#what-is-codegemma)
    - [Evaluation Results](#evaluation-results)
    - [Prompt format](#prompt-format)
- [Using CodeGemma](#using-codegemma)
    - [Using Transformers](#using-transformers)
    - [Integration with Google Cloud](#integration-with-google-cloud)
    - [Integration with Inference Endpoints](#integration-with-inference-endpoints)
- [Additional Resources](#additional-resources)

## What is CodeGemma?

CodeGemma is a family of code-specialist LLM models by Google, based on the pre-trained [2B and 7B Gemma checkpoints](https://huggingface.co/blog/gemma). CodeGemma are further trained on an additional 500 billion tokens of primarily English language data, mathematics, and code to improve on logical and mathematical reasoning, and are suitable for code completion and generation.

[CodeGemma 2B](https://huggingface.co/google/codegemma-2b) was trained exclusively on Code Infilling and is meant for fast code completion and generation, especially in settings where latency and/or privacy are crucial. [CodeGemma 7B](https://huggingface.co/google/codegemma-7b) training mix includes code infilling data (80%) and natural language. It can be used for code completion, as well as code and language understanding and generation. [CodeGemma 7B Instruct](https://huggingface.co/google/codegemma-7b-it) was fine-tuned for instruction following on top of CodeGemma 7B. It’s meant for conversational use, especially around code, programming, or mathematical reasoning topics. All the models have the same 8K token context size as their predecessors.


![The CodeGemma family](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/codegemma/codegemma-family.png "The CodeGemma family")

This image is from [the original report](https://goo.gle/codegemma)

### Evaluation Results

CodeGemma-7B outperforms similarly-sized 7B models except DeepSeek-Coder-7B on HumanEval, a popular benchmark for evaluating code models on Python. The same goes for the evaluation of other programming languages like Java, JavaScript, and C++ from MultiPL-E, a translation of HumanEval. According to the technical report, the model performs best on [GSM8K](https://huggingface.co/datasets/gsm8k) among 7B models. The instruct version CodeGemma-7B-it improves on the most popular languages on both HumanEval and MBPP (cf paper table 5). For more details, you can check the [BigCode leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard) or some metrics below.

| Model | Pretraining size [tokens] | Python | JavaScript |
| --- | --- | --- | --- |
| 10B+ models |  |  |  |
| StarCoder 2 15B | 4,000B+ | 44.15 | 44.24 |
| Code Llama 13B | 2,500B | 35.07 | 38.26 |
| 7B models |  |  |  |
| DeepSeek Coder 7B | 2,000B | 45.83 | 45.9 |
| CodeGemma 7B | 500B of extra training | 40.13 | 43.06 |
| Code Llama 7B | 2,500B | 29.98 | 31.8 |
| StarCoder 2 7B | 3,500B+ | 34.09 | 35.35 |
| StarCoderBase 7B | 3,000B+ | 28.37 | 27.35 |
| <3B models |  |  |  |
| CodeGemma 2B | 500B of extra training | 27.28 | 29.94 |
| Stable Code 3B | 1,300B | 30.72 | 28.75 |
| StarCoder 2 3B | 3,000B+ | 31.44 | 35.37 |

| Model | Pretraining size [tokens] | Python | JavaScript |
| --- | --- | --- | --- |
| 10B+ models |  |  |  |
| Code Llama 13B | 2,620B | 50.6 | 40.92 |
| Code Llama 13B | 2,620B | 42.89 | 40.66 |
| 7B models |  |  |  |
| CodeGemma 7B | 500B | 52.74 | 47.71 |
| Code Llama 7B  | 2,620B | 40.48 | 36.34 |
| Code Llama 7B | 2,620B | 25.65 | 33.11 |

Here is a table from the original report with a breakdown per language.

![CodeGemma quality across languages](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/codegemma/codegemma-family.png "CodeGemma quality across languages")


### Prompt format

CodeGemma 2B and CodeGemma 7B use infilling (code, comments, docstrings, import statements) for code completion. CodeGemma was trained for this task using the fill-in-the-middle (FIM) objective, where you provide a prefix and a suffix as context for the completion. The following tokens are used to separate the different parts of the input:

- `<|fim_prefix|>` precedes the context before the completion we want to run.
- `<|fim_suffix|>` precedes the suffix. You must put this token exactly where the cursor would be positioned in an editor, as this is the location where the model will code complete.
- `<|fim_middle|>` is the prompt that invites the model to run the generation.

In addition to these, there's also `<|file_separator|>`, which provides multi-file contexts. We’ll show examples of use in the *Using with transformers* section.

CodeGemma 7B Instruct uses the same prompt format as the base Gemma Instruction-tuned versions, following this conversation structure:

```bash
<bos><start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
LaMDA<end_of_turn>
<start_of_turn>model
LaMDA who?<end_of_turn>
```

As is the case with Gemma, the easiest way to reproduce this format is with the chat template available in `transformers`.

## Using CodeGemma

### Using Transformers

With Transformers [release 4.39](https://github.com/huggingface/transformers/releases/tag/v4.39.3), you can use CodeGemma and leverage all the tools within the Hugging Face ecosystem, such as:

- training and inference scripts and examples
- safe file format (`safetensors`)
- integrations with tools such as bitsandbytes (4-bit quantization), PEFT (parameter efficient fine-tuning), and Flash Attention 2
- utilities and helpers to run generation with the model
- mechanisms to export the models to deploy

Like the Gemma models, CodeGemma is compatible with `torch.compile()` for an important inference speedup.

To use CodeGemma with transformers, make sure to use the latest release:

```jsx
pip install --upgrade transformers
```

The following snippet shows how to use `codegemma-2b` for code completion with transformers. It requires about 6 GB of RAM using `float16` precision, making it perfectly suitable for consumer GPUs and on-device applications.

```python
from transformers import GemmaTokenizer, AutoModelForCausalLM
import torch

model_id = "google/codegemma-2b"
tokenizer = GemmaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
	model_id,
	torch_dtype=torch.float16
).to("cuda:0")

prompt = '''\
<|fim_prefix|>import datetime
def calculate_age(birth_year):
    """Calculates a person's age based on their birth year."""
    current_year = datetime.date.today().year
    <|fim_suffix|>
    return age<|fim_middle|>\
'''

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
prompt_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][prompt_len:]))
```

Observe that the `<|fim_suffix|>` token appears in the position where the cursor would be placed in an editor, marking the position for the generation. `<|fim_prefix|>` provides the context that precedes the cursor, and the remaining until `<|fim_middle|>` is additional context after the cursor. Either of them can be empty if the cursor is located at the beginning or end of the file.

The previous code may return something like the following:

```
age = current_year - birth_year<|file_separator|>test_calculate_age.py
<|fim_suffix|>
    assert calculate_age(1990) == 33
    assert calculate_age(1980) == 43
    assert calculate_age(1970) == 53
    assert calculate_age(1960) == 63
    assert calculate_age(1950) == 73
```

Note the extra content after the correct completion. This is particularly the case for CodeGemma 7B, which is more verbose and tends to provide additional code or comments after completion. We must ignore everything that appears after the FIM tokens or the EOS token for code infilling. We can stop generation early with transformers by providing a list of terminators to the `generate` function, like this:

```python
FIM_PREFIX = '<|fim_prefix|>'
FIM_SUFFIX = '<|fim_suffix|>'
FIM_MIDDLE = '<|fim_middle|>'
FIM_FILE_SEPARATOR = '<|file_separator|>'

terminators = tokenizer.convert_tokens_to_ids(
	[FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_FILE_SEPARATOR]
)
terminators += [tokenizer.eos_token_id]

outputs = model.generate(
  **inputs,
  max_new_tokens=100,
  eos_token_id=terminators,
)
```

In this case, generation will stop as soon as the first delimiter is found:

```
age = current_year - birth_year<|file_separator|>
```

### A note on precision

The original CodeGemma checkpoints are released in `bfloat16` precision. If you load the model without indicating a `torch_dtype`, PyTorch will upcast them to `float32`. Casting to `float16` is perfectly fine for use, and it can be much faster than `bfloat16` on certain hardware. For maximum precision, we recommend you use `bfloat16` rather than `float32`.

You can also automatically quantize the model, loading it in 8-bit or 4-bit mode. 4-bit loading of CodeGemma 7B takes about 9 GB of memory to run, making it compatible with many consumer cards and all the GPUs in Google Colab. This is how you’d load the generation pipeline in 4-bit:

```jsx
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True}
    },
)
```

### Integration with Google Cloud

You can deploy and train Gemma on Google Cloud through Vertex AI or Google Kubernetes Engine (GKE), using [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index) and Transformers. 

To deploy the CodeGemma model from Hugging Face, go to the [model page](https://huggingface.co/google/codegemma-7b-it) and click on [Deploy -> Google Cloud.](https://huggingface.co/google/codegemma-7b-it) This will bring you to the Google Cloud Console, where you can 1-click deploy CodeGemma on Vertex AI or GKE, powered by Text Generation Inference.

You can also access CodeGemma directly through the Vertex AI Model Garden. 

![GCP Integration](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/codegemma/gcp-integration.png "GCP Integration")

## Integration with Inference Endpoints

You can deploy CodeGemma on Hugging Face's [Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=google/codegemma-2b&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=2xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_total_tokens=1024000&tgi_max_total_tokens=32000), which uses Text Generation Inference as the backend. [Text Generation Inference](https://github.com/huggingface/text-generation-inference) is a production-ready inference container developed by Hugging Face to enable easy deployment of large language models. It has features such as continuous batching, token streaming, tensor parallelism for fast inference on multiple GPUs, production-ready logging and tracing, and is distributed under the Apache 2 license.

To deploy a CodeGemma model, go to the [model page](https://huggingface.co/google/codegemma-2b) and click on the [Deploy -> Inference Endpoints](https://ui.endpoints.huggingface.co/new?repository=google/codegemma-2b) widget. You can learn more about [Deploying LLMs with Hugging Face Inference Endpoints](https://huggingface.co/blog/inference-endpoints-llm) in a previous blog post. Note that T4s do not support the `bfloat16` format, so you will need to use a different GPU option.

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model=IE_ENDPOINT)

prompt = """\
<|fim_prefix|>import <|fim_suffix|>

if __name__ == '__main__':
  sys.exit(0)<|fim_middle|>\
"""

client.text_generation(prompt=prompt)
```

## Additional Resources

- [Models on the Hub](https://huggingface.co/models?search=google/codegemma)
- [Code Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
- [Technical Report](https://goo.gle/codegemma)
