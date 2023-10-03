---
title: "Overview of natively supported quantization schemes in 🤗 Transformers" 
thumbnail: /blog/assets/163_overview_quantization_transformers/thumbnail.jpg
authors:
- user: ybelkada
- user: marcsun13
- user: IlyasMoutawwakil
- user: clefourrier
- user: fxmarty
---

# Overview of natively supported quantization schemes in 🤗 Transformers



We aim to give a clear overview of the pros and cons of each quantization scheme supported in transformers to help you decide which one you should go for.

Currently, quantizing models are used for two main purposes:

- Running inference of a large model on a smaller device
- Fine-tune adapters on top of quantized models 

So far, two integration efforts have been made and are **natively** supported in transformers : *bitsandbytes* and *auto-gptq*.
Note that some additional quantization schemes are also supported in the [🤗 optimum library](https://github.com/huggingface/optimum), but this is out of scope for this blogpost. 

To learn more about each of the supported schemes, please have a look at one of the resources shared below. Please also have a look at the appropriate sections of the documentation.

Note also that the details shared below are only valid for `PyTorch` models, this is currently out of scope for Tensorflow and Flax/JAX models.

## Table of contents

- [Resources](#resources)
- [Comparing bitsandbytes and auto-gptq](#Comparing-bitsandbytes-and-auto-gptq)
- [Diving into speed benchmarks](#Diving-into-speed-benchmarks)
- [Conclusion and final words](#conclusion-and-final-words)
- [Acknowledgements](#acknowledgements)

## Resources

- [GPTQ blogpost](https://huggingface.co/blog/gptq-integration) – gives an overview on what is the GPTQ quantization method and how to use it. 
- [bistandbytes 4-bit quantization blogpost](https://huggingface.co/blog/4bit-transformers-bitsandbytes) - This blogpost introduces 4-bit quantization and QLoRa, an efficient finetuning approach. 
- [bistandbytes 8-bit quantization blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration) - This blogpost explains how 8-bit quantization works with bitsandbytes.
- [Basic usage Google Colab notebook for GPTQ](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) -  This notebook shows how to quantize your transformers model with the GPTQ method, how to do inference, and how to do fine-tuning with the quantized model.
- [Basic usage Google Colab notebook for bitsandbytes](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing) - This notebook shows how to use 4-bit models in inference with all their variants, and how to run GPT-neo-X (a 20B parameter model) on a free Google Colab instance.
- [Merve's blogpost on quantization](https://huggingface.co/blog/merve/quantization) - This blogpost provides a gentle introduction to quantization and the quantization methods supported natively in transformers. 


## Comparing bitsandbytes and auto-gptq
In this section, we will go over the pros and cons of bitsandbytes and gptq quantization. Note that these are based on the feedback from the community and they can evolve over time as some of these features are in the roadmap of the respective libraries.

### What are the benefits of bitsandbytes?
**easy**: bitsandbytes still remains the easiest way to quantize any model as it does not require calibrating the quantized model with input data (also called zero-shot quantization). It is possible to quantize any model out of the box as long as it contains `torch.nn.Linear` modules. Whenever a new architecture is added in transformers, as long as they can be loaded with accelerate’s `device_map=”auto”`, users can benefit from bitsandbytes quantization straight out of the box with minimal performance degradation. Quantization is performed on model load, no need to run any post-processing or preparation step.

**cross-modality interoperability**: As the only condition to quantize a model is to contain a `torch.nn.Linear` layer, quantization works out of the box for any modality, making it possible to load models such as Whisper, ViT, Blip2, etc. in 8-bit or 4-bit out of the box.

**0 performance degradation when merging adapters**: (Read more about adapters and PEFT in [this blogpost](https://huggingface.co/blog/peft) if you are not familiar with it). If you train adapters on top of the quantized base model, the adapters can be merged on top of of the base model for deployment, with no inference performance degradation. You can also [merge](https://github.com/huggingface/peft/pull/851/files) the adapters on top of the dequantized model ! This is not supported for GPTQ. 


### What are the benefits of autoGPTQ?
**fast for text generation**: GPTQ quantized models are fast compared to bitsandbytes quantized models for [text generation](https://huggingface.co/docs/transformers/main_classes/text_generation). We will address the speed comparison in an appropriate section. 

**n-bit support**: The GPTQ algorithm makes it possible to quantize models up to 2 bits! However, this might come with severe quality degradation. The recommended number of bits is 4, which seems to be a great tradeoff for GPTQ at this time.

**easily-serializable**: GPTQ models support serialization for any number of bits. Loading models from TheBloke namespace: https://huggingface.co/TheBloke (look for those that end with the `-GPTQ` suffix) is supported out of the box, as long as you have the required packages installed. Bitsandbytes supports 8-bit serialization but does not support 4-bit serialization as of today.

**AMD support**: The integration should work out of the box for AMD GPUs!

### What are the potential rooms of improvements of bitsandbytes?
**slower than GPTQ for text generation**: bitsandbytes 4-bit models are slow compared to GPTQ when using [`generate`](https://huggingface.co/docs/transformers/main_classes/text_generation).

**4-bit weights are not serializable**: Currently, 4-bit models cannot be serialized. This is a frequent community request, and we believe it should be addressed very soon by the bitsandbytes maintainers as it's in their roadmap! 

### What are the potential rooms of improvements of autoGPTQ?
**calibration dataset**: The need of a calibration dataset might discourage some users to go for GPTQ. Furthermore, it can take several hours to quantize the model (e.g. 4 GPU hours for a 175B scale model [according to the paper](https://arxiv.org/pdf/2210.17323.pdf) - section 2)

**works only for language models (for now)**: As of today, the API for quantizing a model with auto-GPTQ has been designed to support only language models. It should be possible to quantize non-text (or multimodal) models using the GPTQ algorithm, but the process has not been elaborated in the original paper or in the auto-gptq repository. If the community is excited about this topic this might be considered in the future.

## Diving into speed benchmarks 
We decided to provide an extensive benchmark for both inference and fine-tuning adapters using bitsandbytes and auto-gptq on different hardware. The inference benchmark should give users an idea of the speed difference they might get between the different approaches we propose for inference, and the adapter fine-tuning benchmark should give a clear idea to users when it comes to deciding which approach to use when fine-tuning adapters on top of bitsandbytes and GPTQ base models.

We will use the following setup: 
- bitsandbytes: 4-bit quantization with `bnb_4bit_compute_dtype=torch.float16`. Make sure to use `bitsandbytes>=0.41.1` for fast 4-bit kernels. 
- auto-gptq: 4-bit quantization with exllama kernels. You will need `auto-gptq>=0.4.0` to use ex-llama kernels. 

### Inference speed (forward pass only)

This benchmark measures only the prefill step, which corresponds to the forward pass during training. It  was run on a single NVIDIA A100-SXM4-80GB GPU with a prompt length of 512. The model we used was `meta-llama/Llama-2-13b-hf`.

with batch size = 1: 

|quantization |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|fp16|None     |None|None      |None  |26.0         |36.958                |27.058            |29152.98        |
|gptq |False    |4   |128       |exllama|36.2         |33.711                |29.663            |10484.34        |
|bitsandbytes|None     |4|None      |None  |37.64        |52.00                 |19.23             |11018.36       |

with batch size = 16:

|quantization |act_order|bits|group_size|kernel|Load time (s)|Per-token latency (ms)|Throughput (tok/s)|Peak memory (MB)|
|-----|---------|----|----------|------|-------------|----------------------|------------------|----------------|
|fp16|None     |None|None      |None  |26.0         |69.94                 |228.76            |53986.51        |
|gptq |False    |4   |128       |exllama|36.2         |95.41                 |167.68            |34777.04        |
|bitsandbytes|None     |4|None      |None  |37.64        |113.98                |140.38            |35532.37       |

From the benchmark, we can see that bitsandbyes and GPTQ are equivalent, with GPTQ being slightly faster for large batch size. Check this [link](https://github.com/huggingface/optimum/blob/main/tests/benchmark/README.md#prefill-only-benchmark-results) to have more details on these benchmarks.  

### Generate speed

The following benchmarks measure the generation speed of the model during inference. The benchmarking script can be found [here](https://gist.github.com/younesbelkada/e576c0d5047c0c3f65b10944bc4c651c) for reproducibility.

#### use_cache 
Let's test `use_cache` to better understand the impact of caching the hidden state during the generation.

The benchmark was run on an A100 with a prompt length of 30 and we generated exactly 30 tokens. The model we used was `meta-llama/Llama-2-7b-hf`. 

with `use_cache=True`

![Benchmark use_cache=True A100](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/A100_use_cache_True.jpg)

with `use_cache=False`

![Benchmark use_cache=False A100](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/A100_use_cache_False.jpg)

From the two benchmarks, we conclude that generation is faster when we use attention caching, as expected. Moreover, GPTQ is, in general, faster than bitsandbytes. For example, with `batch_size=4` and `use_cache=True`, it is twice as fast! Therefore let’s use `use_cache` for the next benchmarks. Note that `use_cache` will consume more memory. 

#### Hardware

In the following benchmark, we will try different hardware to see the impact on the quantized model. We used a prompt length of 30 and we generated exactly 30 tokens. The model we used was `meta-llama/Llama-2-7b-hf`.

with a NVIDIA A100: 

![Benchmark A100](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/A100_use_cache_True.jpg)

with a NVIDIA T4: 

![Benchmark T4](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/T4.jpg)

with a Titan RTX: 

![Benchmark TITAN RTX](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/RTX_Titan.jpg)

From the benchmark above, we can conclude that GPTQ is faster than bitsandbytes for those three GPUs. 

#### Generation length 

In the following benchmark, we will try different generation lengths to see their impact on the quantized model. It was run on a A100 and we used a prompt length of 30, and varied the number of generated tokens. The model we used was `meta-llama/Llama-2-7b-hf`.

with 30 tokens generated:

![Benchmark A100](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/A100_use_cache_True.jpg)

with 512 tokens generated:

![Benchmark A100 512 tokens](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/A100_max_token_512.jpg)

From the benchmark above, we can conclude that GPTQ is faster than bitsandbytes independently of the generation length. 

### Adapter fine-tuning (forward + backward)

It is not possible to perform pure training on a quantized model. However, you can fine-tune quantized models by leveraging parameter efficient fine tuning methods (PEFT) and train adapters on top of them. The fine-tuning method will rely on a recent method called "Low Rank Adapters" (LoRA): instead of fine-tuning the entire model you just have to fine-tune these adapters and load them properly inside the model. Let's compare the fine-tuning speed! 

The benchmark was run on a NVIDIA A100 GPU and we used `meta-llama/Llama-2-7b-hf` model from the Hub. Note that for GPTQ model, we had to disable the exllama kernels as exllama is not supported for fine-tuning.

![Benchmark A100 finetuning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/163_overview-quantization-transformers/A100_finetuning.png)

From the result, we conclude that bitsandbytes is faster than GPTQ for fine-tuning. 

### Performance degradation

Quantization is great for reducing memory consumption. However, it  does come with performance degradation. Let's compare the performance using the [Open-LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) ! 

with 7b model: 

| model_id                           | Average | ARC   | Hellaswag | MMLU  | TruthfulQA |
|------------------------------------|---------|-------|-----------|-------|------------|
| meta-llama/llama-2-7b-hf           | **54.32**   | 53.07 | 78.59     | 46.87 | 38.76      |
| meta-llama/llama-2-7b-hf-bnb-4bit  | **53.4**    | 53.07 | 77.74     | 43.8  | 38.98      |
| TheBloke/Llama-2-7B-GPTQ           | **53.23**   | 52.05 | 77.59     | 43.99 | 39.32      |

with 13b model: 

| model_id                           | Average | ARC   | Hellaswag | MMLU  | TruthfulQA |
|------------------------------------|---------|-------|-----------|-------|------------|
| meta-llama/llama-2-13b-hf          | **58.66**   | 59.39 | 82.13     | 55.74 | 37.38      |
| TheBloke/Llama-2-13B-GPTQ (revision = 'gptq-4bit-128g-actorder_True')| **58.03**   | 59.13 | 81.48     | 54.45 | 37.07      |
| TheBloke/Llama-2-13B-GPTQ          | **57.56**   | 57.25 | 81.66     | 54.81 | 36.56      |
| meta-llama/llama-2-13b-hf-bnb-4bit | **56.9**    | 58.11 | 80.97     | 54.34 | 34.17      |

From the results above, we conclude that there is less degradation in bigger models. More interestingly, the degradation is minimal!

## Conclusion and final words

In this blogpost, we compared bitsandbytes and GPTQ quantization across multiple setups. We saw that bitsandbytes is better suited for fine-tuning while GPTQ is better for generation. From this observation, one way to get better merged models would be to: 

- (1) quantize the base model using bitsandbytes (zero-shot quantization)
- (2) add and fine-tune the adapters
- (3) merge the trained adapters on top of the base model or the [dequantized model](https://github.com/huggingface/peft/pull/851/files) !
- (4) quantize the merged model using GPTQ and use it for deployment 

We hope that this overview will make it easier for everyone to use LLMs in their applications and usecases, and we are looking forward to seeing what you will build with it!

## Acknowledgements

We would like to thank [Ilyas](https://huggingface.co/IlyasMoutawwakil), [Clémentine](https://huggingface.co/clefourrier) and [Felix](https://huggingface.co/fxmarty) for their help on the benchmarking. 

Finally, we would like to thank [Pedro Cuenca](https://github.com/pcuenca) for his help with the writing of this blogpost.
