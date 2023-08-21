---
title: "Llama 2 is here - get it on Hugging Face" 
thumbnail: /blog/assets/159_autogptq_transformers/thumbnail.jpg
authors:
- user: marcsun13
- user: fxmarty
- user: ybelkada
- user: TheBloke
---

# Making the LLMs lighter with AutoGPTQ, transformers and optimum

Large language model have demonstrated remarkable capabilities in understanding and generating human-like text, revolutionizing applications across various domains. However, as these LLMs continue to grow in size and complexity, the demands they place on consumer hardware for training and deployment have become increasingly challenging to meet. 

At Hugging Face 🤗, we strive to make large models as accessible as possible for everyone. In the same spirit as the bitsandbytes collaboration, we integrated in Transformers with the [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) library to allow users to quantize and run their models in 8, 4, 3 or even 2-bit precision using GPTQ algorithm ([Frantar et al. 2023](https://arxiv.org/pdf/2210.17323.pdf)). There is negligible accuracy degradation when quantizating down to 4-bit precision, with comparable inference speed over FP16 for small batch sizes.

This integration is available both for Nvidia GPUs, and RoCm-powered AMD GPUs.

## Ressources

This blogpost and release come with several resources to get started with GPTQ quantization:

- [Original Paper](https://arxiv.org/pdf/2210.17323.pdf)
- Basic usage Google Colab notebook -  This notebook shows how to quantize your transformers model with GPTQ method, how to do inference, and how to do fine-tuning with the quantized model.
- Transformers integration [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
- Optimum integration [documentation](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- The Bloke [repository](https://huggingface.co/TheBloke?sort_models=likes#models) with compatible GPTQ models.


## **A gentle summary of GPTQ paper**

Quantization methods are mostly contained into two categories: 

1. Post-Training Quantization (PTQ): We quantize a pre-trained model using low resources, such as a few samples and a few hours of computation. 
2. Quantization-Aware Training (QAT): Quantization is performed before training or further fine-tuning. 

The GPTQ falls into the PTQ category and this is particularly interesting for massive models, for which full model training or even fine-tuning can be very expensive.

Specifically, GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16. During inference, weights are dequantized on the fly and the actual compute is performed in float16.

The benefits of this scheme are twofold:

- Memory savings close to x4 for int4 quantization, as the dequantization happens iteratively close to the compute unit in a fused kernel, and not in the GPU global memory.
- Potential speedups thanks to the time saved on data communication due to the lower bitwidth used for weights.

The GPTQ paper tackles the layer-wise compression problem: 

Given a layer $l$ with weight matrix $W_{l}$ and layer input $X_{l}$, we want to find a quantized version of the weight $\hat{W}_{l}$ to minimize the mean squared error (MSE):

${\hat{W}_{l}}^{*} = argmin_{\hat{W_{l}}} \|W_{l}X-\hat{W}_{l}X\|^{2}_{2}$

Once this is solved per layer, a solution to the global problem can be obtained by combining the layer-wise solutions. 

In order to solve this layer-wise compression problem, the author uses the Optimal Brain Quantization framework ([Frantar et al 2022](https://arxiv.org/abs/2208.11580)). The OBQ method starts from the observation that the above equation can be written as the sum of the squared errors, over each row of $W_{l}$.

$ \sum_{i=0}^{d_{row}} \|W_{l[i,:]}X-\hat{W}_{l[i,:]}X\|^{2}_{2} $

This means that we can quantize each row independently. This is called per-channel quantization. For each row $W_{l[i,:]}$, OBQ quantize one weight at a time while always updating all not-yet-quantized weights, in order to compensate for the error incurred by quantizing a single weight. The update on selected weights has a closed-form formula, utilizing Hessian matrices. 

The GPTQ paper improves this framework by introducing a set of optimization that reduces the complexity of the quantization algorithm while retaining the accuracy of the model.

Compared to OBQ, the quantization step itself is also faster with GPTQ: it takes 2 GPU-hours to quantize a BERT model (336M) with OBQ, whereas with GPTQ, a Bloom model (176B) can be quantized in less than 4 GPU-hours. 

To learn more about the exact algorithm and the different benchmarks on perplexity and speedups, check out the original [paper](https://arxiv.org/pdf/2210.17323.pdf).

## **AutoGPTQ library - the one-stop library for efficiently leveraging GPTQ method for LLMs**

The AutoGPTQ library enables users to quantize 🤗 Transformers models using the GPTQ method. While parallel community efforts as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [Exllama](https://github.com/turboderp/exllama) and [llama.cpp](https://github.com/ggerganov/llama.cpp/) implement quantization methods as GPTQ with optimizations strictly for Llama, AutoGPTQ gained popularity through its smooth coverage of a wide range of transformers architectures.

Thus, to make running large models through transformers even more accessible, we decided to provide an API natively in 🤗 Transformers to quantize and run models using AutoGPTQ. For now, we only integrated the most common optimization options such as CUDA kernels. For more advanced options such as Triton kernels or fused-attention compatibility, check out directly the [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) library.

## Native support of GPTQ models in 🤗 Transformers

After [installing AutoGPTQ library](https://github.com/PanQiWei/AutoGPTQ#quick-installation) and optimum library, running GPTQ models in Transformers is now as simple as:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", torch_dtype=torch.float16, device_map="auto")
```

Check out the Transformers [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) to learn more about all the features. 

This integration through AutoGPTQ has many advantages:

- Quantized models are serializable and can be shared on the Hub.
- GPTQ drastically reduces the memory requirements to run LLMs, while the inference latency is on par with FP16 inference.
- AutoGPTQ supports Exllama kernels for a wide range of architectures.
- The integration comes with native RoCm support for AMD GPUs.
- Finetuning with PEFT is available.

You can check on the Hub if your favorite model has already been quantized. TheBloke, one of Hugging Face top contributor, has quantized a lot of models with AutoGPTQ library and shared them on the Hugging Face Hub. We worked together to make sure that these repositories will work out of the box with our integration.

Below is a sample of benchmark for the batch size = 1 case. The benchmark is done on a single NVIDIA A100-SXM4-80GB GPU. We use a prompt length of 512, and generate exactly 512 new tokens.

| gptq  | act_order | bits | group_size | kernel            | Load time (s) | Per-token latency (ms) | Throughput (tokens/s) | Peak memory (MB) |
|-------|-----------|------|------------|-------------------|---------------|------------------------|-----------------------|------------------|
| False | None      | None | None       | None              | 26.0          | 36.958                 | 27.058                | 29152.98         |
| True  | False     | 4    | 128        | exllama           | 36.2          | 33.711                 | 29.663                | 10484.34         |
| True  | False     | 4    | 128        | autogptq-cuda-old | 36.2          | 46.44                  | 21.53                 | 10344.62         |

A more comprehensive reproducible benchmark is available [here](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark).


## Quantizing models **through Optimum library**

To integrate seemingly AutoGPTQ into Transformers library, we exposed a minimalist version of the AutoGPTQ API in [Optimum library](https://github.com/huggingface/optimum), the Hugging Face toolkit for training and inference optimization. By doing so, it is easy to integrate with Transformers through this external library, while letting people use the Optimum API if they want to quantize their own models! Check out the Optimum [documentation](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization) if you want to quantize your own LLMs. 

Quantizing 🤗 Transformers models with GPTQ method can be done in a few lines:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
```

One major downside is that quantizing a model takes a long time. Note that for a 175B model, at least 4 GPU-hours are required. However, as mentioned above, many GPTQ models are already available on the Hugging Face Hub.

## Running GPTQ models through ***Text-Generation-Inference***

In parallel to the integration of GPTQ in Transformers, GPTQ support was added in the [Text-Generation-Inference library](https://github.com/huggingface/text-generation-inference) (TGI), aimed at serving large language models with features as dynamic batching, paged attention and flash attention, that can now be used along GPTQ for a [wide range of architectures](https://huggingface.co/docs/text-generation-inference/main/en/supported_models).

This integration allows for example to serve a 70b model on a single A100-80GB GPU, which is not possible using a fp16 checkpoint that exceeds the GPU available memory.

You can find out more about the usage of GPTQ in TGI in [the documentation](https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/preparing_model#quantization).

Note that the kernel integrated in TGI does not scale very well with larger batch sizes, and although the approach saves memory, slowdowns are expected at larger batch size.

## **Fine-tune quantized models with PEFT**

You can not train native quantized model. However, by leveraging the PEFT library, you can for example train adapters on top of them. To do that, we freeze all the layers of the quantized model and we add the adapters that are trainable. Here are some examples on how to use PEFT with GPTQ model : [colab notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) and [finetuning](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) script. 

## Room for improvement

As is, GPTQ already brings impressive benefits at a small cost in the quality of prediction. There are however still room for improvement, both in the quantization technique and kernel implementation.

First, while AutoGPTQ integrates (to the best of our knowledge) with the most performant W4A16 kernel (weights as int4, activations as fp16) namely from the [exllama implementation](https://github.com/turboderp/exllama), there is good chance the kernel can still be improved. There have been other promising implementations [from Kim et al.](https://arxiv.org/pdf/2211.10017.pdf) and from [MIT Han Lab](https://github.com/mit-han-lab/llm-awq) that appear to be promising. Moreover, from internal benchmarks, there appears to still be no open-source performant W4A16 kernel written in Triton, which could be a direction to explore.

On the quantization side, let’s emphasize again that this method quantizes only the weights. There have been other approaches proposed for LLM quantization that can quantize both weights and activations at a small cost in prediction quality, as [LLM-QAT](https://arxiv.org/pdf/2305.17888.pdf) where a mixed int4/int8 scheme can be used, as well as quantization of the key-value cache. One of the strong advantage of such technique is the ability to use actual integer arithmetic for the compute, with e.g. [Nvidia Tensor Cores supporting int8 compute](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). However, to the best of our knowledge, there are no open-source W4A8 quantization kernel available, but this may well be [an interesting direction to explore](https://www.qualcomm.com/news/onq/2023/04/floating-point-arithmetic-for-ai-inference-hit-or-miss).

On the kernel side as well, as the benchmarks above show, it remains open to design performant W4A16 kernel for larger batch sizes.

### Supported models

For now, only large language models with a decoder or encoder only architecture are supported. Even if it seems a little bit restrictive, it encompasses most of the state of art LLMs such as Llama, OPT, GPT-Neo, GPT-NeoX.

Support for very large vision, audio, and multi-modal models are currently not supported yet.