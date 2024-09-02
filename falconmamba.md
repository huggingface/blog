---
title: "Welcome Falcon Mamba: The first strong attention-free 7B model" 
thumbnail: /blog/assets/falconmamba/thumbnail.png
authors:
- user: JingweiZuo
  guest: true
  org: tiiuae
- user: yellowvm
  guest: true
  org: tiiuae
- user: DhiyaEddine
  guest: true
  org: tiiuae
- user: IChahed
  guest: true
  org: tiiuae
- user: ybelkada
  guest: true
  org: tiiuae
- user: Gkunsch
  guest: true
  org: tiiuae
---

[Falcon Mamba](https://falconllm.tii.ae/tii-releases-first-sslm-with-falcon-mamba-7b.html) is a new model by [Technology Innovation Institute (TII)](https://www.tii.ae/ai-and-digital-science) in Abu Dhabi released under the [TII Falcon Mamba 7B License 1.0](https://falconllm.tii.ae/falcon-mamba-7b-terms-and-conditions.html). The model is open access and available within the Hugging Face ecosystem [here](https://huggingface.co/tiiuae/falcon-mamba-7b) for anyone to use for their research or application purposes.

In this blog, we will go through the design decisions behind the model, how the model is competitive with respect to other existing SoTA models, and how to use it within the Hugging Face ecosystem.

## First general purpose large-scale pure Mamba model

Transformers, based on the attention mechanism, are the dominant architecture used in all the strongest large language models today. Yet, the attention mechanism is fundamentally limited in processing large sequences due to the increase in compute and memory costs with sequence length. Various alternative architectures, in particular State Space Language Models (SSLMs), tried to address the sequence scaling limitation but fell back in performance compared to SoTA transformers.

With Falcon Mamba, we demonstrate that sequence scaling limitation can indeed be overcome without loss in performance. Falcon Mamba is based on the original Mamba architecture, proposed in [*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/abs/2312.00752), with the addition of extra RMS normalization layers to ensure stable training at scale. This choice of architecture ensures that Falcon Mamba:
* can process sequences of arbitrary length without any increase in memory storage, in particular, fitting on a single A10 24GB GPU.
* takes a constant amount of time to generate a new token, regardless of the size of the context (see this [section](#hardware-performance))

## Model training

Falcon Mamba was trained with ~ 5500GT of data, mainly composed of RefinedWeb data with addition of high-quality technical data and code data from public sources. We used constant learning rate for the most of the training, followed by a relatively short learning rate decay stage. In this last stage, we also added a small portion of high-quality curated data to further enhance model performance.

## Evaluations


We evaluate our model on all benchmarks of the new leaderboard's version using the `lm-evaluation-harness` package and then normalize the evaluation results with Hugging Face score normalization.


| `model name`              |`IFEval`| `BBH` |`MATH LvL5`| `GPQA`| `MUSR`|`MMLU-PRO`|`Average`| 
|:--------------------------|:------:|:-----:|:---------:|:-----:|:-----:|:--------:|:-------:|
| ***Pure SSM models***     |        |       |           |       |       |          |         |
| `Falcon Mamba-7B`          | 33.36  | 19.88 |    3.63   | 8.05  | 10.86 | 14.47    |**15.04**|
| `TRI-ML/mamba-7b-rw`<sup>*</sup>| 22.46  | 6.71  | 0.45      | 1.12  | 5.51  | 1.69     | 6.25    |
|***Hybrid SSM-attention models***   |       |           |       |       |          |         |
|`recurrentgemma-9b`        | 30.76  | 14.80 | 4.83      | 4.70  | 6.60  | 17.88    |  13.20  |
| `Zyphra/Zamba-7B-v1`<sup>*</sup>      | 24.06  | 21.12 | 3.32      | 3.03  | 7.74  | 16.02    | 12.55   |
|***Transformer models***   |        |       |           |       |       |          |         |
| `Falcon2-11B`             | 32.61  | 21.94 |    2.34   | 2.80  | 7.53  | 15.44    |  13.78  |
| `Meta-Llama-3-8B`         | 14.55  | 24.50 |    3.25   | 7.38  | 6.24  | 24.55    |  13.41  |
| `Meta-Llama-3.1-8B`       | 12.70  | 25.29 |    4.61   | 6.15  | 8.98  | 24.95    |  13.78  |
| `Mistral-7B-v0.1`         | 23.86  | 22.02 |    2.49   | 5.59  | 10.68 | 22.36    |  14.50  |
| `Mistral-Nemo-Base-2407 (12B)`       | 16.83  | 29.37 |    4.98   | 5.82  | 6.52  | 27.46    |  15.08  |
| `gemma-7B`                | 26.59  | 21.12 |    6.42   | 4.92  | 10.98 | 21.64    |**15.28**|


Also, we evaluate our model on the benchmarks of the first version of the LLM Leaderboard using `lighteval`.


| `model name`                 |`ARC`|`HellaSwag`   |`MMLU` |`Winogrande`|`TruthfulQA`|`GSM8K`|`Average`         | 
|:-----------------------------|:------:|:---------:|:-----:|:----------:|:----------:|:-----:|:----------------:|
| ***Pure SSM models***        |        |           |       |            |            |       |                  |
| `Falcon Mamba-7B`<sup>*</sup>             |62.03   |   80.82   | 62.11 |   73.64    |   53.42    | 52.54 |  **64.09**       |
| `TRI-ML/mamba-7b-rw`<sup>*</sup>         | 51.25  | 80.85     | 33.41 | 71.11      | 32.08      | 4.70  | 45.52            |
|***Hybrid SSM-attention models***|     |           |       |            |            |       |                  |
| `recurrentgemma-9b`<sup>**</sup>          |52.00   |   80.40   | 60.50 |   73.60    |   38.60    | 42.60 |  57.95           |
| `Zyphra/Zamba-7B-v1`<sup>*</sup>         | 56.14  | 82.23     | 58.11 | 79.87      | 52.88      | 30.78 |  60.00           |
|***Transformer models***      |        |           |       |            |            |       |                  |
| `Falcon2-11B`                | 59.73  | 82.91     | 58.37 | 78.30      | 52.56      | 53.83 | **64.28**        |
| `Meta-Llama-3-8B`            | 60.24  | 82.23     | 66.70 | 78.45      | 42.93      | 45.19 | 62.62            |
| `Meta-Llama-3.1-8B`          | 58.53  | 82.13     | 66.43 | 74.35      | 44.29      | 47.92 | 62.28            |
| `Mistral-7B-v0.1`            | 59.98  | 83.31     | 64.16 | 78.37      | 42.15      | 37.83 | 60.97            |
| `gemma-7B`                   | 61.09  |   82.20   | 64.56 |   79.01    |   44.79    | 50.87 |  63.75           |

For the models marked by *star*, we evaluated the tasks internally, while for the models marked by two *stars*, the results were taken from paper or model card.

## Processing large sequences

Following theoretical efficiency SSM models in processing large sequences, we perform a comparison of memory usage and generation throughput between Falcon Mamba and popular transfomer models using the
[optimum-benchmark](https://github.com/huggingface/optimum-benchmark) library. For a fair comparison, we rescaled the vocabulary size of all transformer models to match Falcon Mamba since it has a big impact on the memory requirements of the model.

Before going to the results, let's first discuss the difference between the prompt (prefill) and generated (decode) parts of the sequence. As we will see, the details of prefill are more important for state space models than for transformer models. When a transformer generates the next token, it needs to attend to the keys and values of all previous tokens in the context. This implies linear scaling of both memory requirements and generation time with context length. A state space model attends to and stores only its recurrent state and, therefore, doesn't need additional memory or time to generate large sequences. While this explains the claimed advantage of SSMs over transformers in the decode stage, the prefill stage requires additional effort to fully utilize SSM architecture.

A standard approach for prefill is to process the whole prompt in parallel to fully utilize GPU. This approach is used in [optimum-benchmark](https://github.com/huggingface/optimum-benchmark) library and we will refer to it as parallel prefill. Parallel prefill needs to store in memory the hidden states of each token in the prompt. For transformers, this additional memory is dominated by the memory of stored KV caches. For SSM models, no caching is required, and the memory for storing hidden states becomes the only component proportional to the prompt length. As a result, the memory requirement will scale with prompt length, and SSM models will lose the ability to process arbitrary long sequences, similar to transformers. 

An alternative to parallel prefill is to process the prompt token by token, which we will refer to as *sequential prefill*. Akin to sequence parallelism, it can also be done on larger chunks of the prompt instead of individual tokens for better GPU usage. While sequential prefill makes little sense for transformers, it brings back the possibility of processing arbitrary long prompts by SSM models. 

With these remarks in mind, we first test the largest sequence length that can fit on a single 24 GB A10 GPU, putting the results on the [figure](#max-length) below. The batch size is fixed at 1, and we are using float32 precision. Even for parallel prefill, Falcon Mamba can fit larger sequences than a transformer, while in sequential prefill, it unlocks its full potential and can process arbitrary long prompt

<a id="max-length"></a>
![Model Performance](https://huggingface.co/datasets/tiiuae/documentation-images/resolve/main/falcon_mamba/max_len_llalma3-1.png)

Next, we measure the generation throughput in a setting with a prompt of length 1 and up to 130k generated tokens, using batch size 1 and H100 GPU. The results are reported in the [figure](#throughput) below. We observe that our Falcon Mamba is generating all the tokens at constant throughput and without any increase in CUDA peak memory. For the transformer model, the peak memory grows, and generation speed slows down as the number of generated tokens grows.

<a id="throughput"></a>
![Model Performance](https://huggingface.co/datasets/tiiuae/documentation-images/resolve/main/falcon_mamba/thoughput-llama3-1.png)

## How to use it within Hugging Face transformers?

The Falcon Mamba architecture will be available in the next release of the Hugging Face transformers library (>4.45.0). To use the model, make sure to install the latest version of Hugging Face transformers or install the library from the source.

Falcon Mamba is compatible with most of the APIs Hugging Face offers, which you are familiar with, such as `AutoModelForCausalLM` or `pipeline` : 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer 

model_id = "tiiuae/falcon-mamba-7b" 
tokenizer = AutoTokenizer.from_pretrained(model_id) 

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto") 
inputs = tokenizer("Hello world, today", return_tensors="pt").to(0) 

output = model.generate(**inputs, max_new_tokens=100, do_sample=True) 
print(tokenizer.decode(Output[0], skip_special_tokens=True)) 
```

As the model is large, it also supports features such as `bitsandbytes` quantization to run the model on smaller GPU memory constraints, e.g.: 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig 

model_id = "tiiuae/falcon-mamba-7b" 
tokenizer = AutoTokenizer.from_pretrained(model_id) 

quantization_config = BitsAndBytesConfig(load_in_4bit=True) 
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config) 

inputs = tokenizer("Hello world, today", return_tensors="pt").to(0) 
output = model.generate(**inputs, max_new_tokens=100, do_sample=True) 

print(tokenizer.decode(output[0], skip_special_tokens=True)) 
```
We are also pleased to introduce the instruction-tuned version of Falcon Mamba, which has been fine-tuned with an additional 5 billion tokens of supervised fine-tuning (SFT) data. This extended training enhances the model's ability to perform instructional tasks with better precision and effectiveness. You can experience the capabilities of the instruct model through our demo, available [here](https://huggingface.co/spaces/tiiuae/falcon-mamba-playground). For the chat template we use the following format:
```bash
<|im_start|>user
prompt<|im_end|>
<|im_start|>assistant
```

You can also directly use the 4-bit converted version of both the [base model](https://huggingface.co/tiiuae/falcon-mamba-7b-4bit) and the [instruct model](https://huggingface.co/tiiuae/falcon-mamba-7b-instruct-4bit). Make sure to have access to a GPU that is compatible with `bitsandbytes` library to run the quantized model.

You can also benefit from faster inference using `torch.compile`; simply call `model = torch.compile(model)` once you have loaded the model. 

## Acknowledgments

The authors of this blog post would like to thank the Hugging Face team for their smooth support and integration within their ecosystem, in particular

- [Alina Lozovskaya](https://huggingface.co/alozowski) and [Clementine Fourrier](https://huggingface.co/clefourrier) for helping us evaluating the model on the leaderboard
- [Arthur Zucker](https://huggingface.co/ArthurZ) for the transformers integration
- [Vaibhav Srivastav](https://huggingface.co/reach-vb), [hysts](https://huggingface.co/hysts) and [Omar Sanseviero](https://huggingface.co/osanseviero) for their support with questions related to Hub

The authors would also like to thank Tri Dao and Albert Gu for implementing and open-sourcing Mamba architecture to the community.
