---
title: "Making LLMs lighter with AutoGPTQ and transformers" 
thumbnail: /blog/assets/159_autogptq_transformers/thumbnail.jpg
authors:
- user: marcsun13
- user: fxmarty
- user: PanEa
  guest: true
- user: qwopqwop
  guest: true
- user: ybelkada
- user: TheBloke
  guest: true
---

# Making LLMs lighter with AutoGPTQ and transformers


Large language models have demonstrated remarkable capabilities in understanding and generating human-like text, revolutionizing applications across various domains. However, the demands they place on consumer hardware for training and deployment have become increasingly challenging to meet. 

🤗 Hugging Face's core mission is to _democratize good machine learning_, and this includes making large models as accessible as possible for everyone. In the same spirit as our [bitsandbytes collaboration](https://huggingface.co/blog/4bit-transformers-bitsandbytes), we have just integrated the [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) library in Transformers, making it possible for users to quantize and run models in 8, 4, 3, or even 2-bit precision using the GPTQ algorithm ([Frantar et al. 2023](https://arxiv.org/pdf/2210.17323.pdf)). There is negligible accuracy degradation with 4-bit quantization, with inference speed comparable to the `fp16` baseline for small batch sizes. Note that GPTQ method slightly differs from post-training quantization methods proposed by bitsandbytes as it requires to pass a calibration dataset.

This integration is available both for Nvidia GPUs, and RoCm-powered AMD GPUs.

## Table of contents

- [Resources](#resources)
- [**A gentle summary of the GPTQ paper**](#a-gentle-summary-of-the-gptq-paper)
- [AutoGPTQ library – the one-stop library for efficiently leveraging GPTQ for LLMs](#autogptq-library--the-one-stop-library-for-efficiently-leveraging-gptq-for-llms)
- [Native support of GPTQ models in 🤗 Transformers](#native-support-of-gptq-models-in-🤗-transformers)
- [Quantizing models **with the Optimum library**](#quantizing-models-with-the-optimum-library)
- [Running GPTQ models through ***Text-Generation-Inference***](#running-gptq-models-through-text-generation-inference)
- [**Fine-tune quantized models with PEFT**](#fine-tune-quantized-models-with-peft)
- [Room for improvement](#room-for-improvement)
  * [Supported models](#supported-models)
- [Conclusion and final words](#conclusion-and-final-words)
- [Acknowledgements](#acknowledgements)


## Resources

This blogpost and release come with several resources to get started with GPTQ quantization:

- [Original Paper](https://arxiv.org/pdf/2210.17323.pdf)
- [Basic usage Google Colab notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) -  This notebook shows how to quantize your transformers model with GPTQ method, how to do inference, and how to do fine-tuning with the quantized model.
- Transformers integration [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
- Optimum integration [documentation](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- The Bloke [repositories](https://huggingface.co/TheBloke?sort_models=likes#models) with compatible GPTQ models.


## **A gentle summary of the GPTQ paper**

Quantization methods usually belong to one of two categories: 

1. Post-Training Quantization (PTQ): We quantize a pre-trained model using moderate resources, such as a calibration dataset and a few hours of computation.
2. Quantization-Aware Training (QAT): Quantization is performed before training or further fine-tuning. 

GPTQ falls into the PTQ category and this is particularly interesting for massive models, for which full model training or even fine-tuning can be very expensive.

Specifically, GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16. During inference, weights are dequantized on the fly and the actual compute is performed in float16.

The benefits of this scheme are twofold:

- Memory savings close to x4 for int4 quantization, as the dequantization happens close to the compute unit in a fused kernel, and not in the GPU global memory.
- Potential speedups thanks to the time saved on data communication due to the lower bitwidth used for weights.

The GPTQ paper tackles the layer-wise compression problem: 

Given a layer \\(l\\) with weight matrix \\(W_{l}\\) and layer input \\(X_{l}\\), we want to find a quantized version of the weight \\(\hat{W}_{l}\\) to minimize the mean squared error (MSE):


\\({\hat{W}_{l}}^{*} = argmin_{\hat{W_{l}}} \|W_{l}X-\hat{W}_{l}X\|^{2}_{2}\\)

Once this is solved per layer, a solution to the global problem can be obtained by combining the layer-wise solutions. 

In order to solve this layer-wise compression problem, the author uses the Optimal Brain Quantization framework ([Frantar et al 2022](https://arxiv.org/abs/2208.11580)). The OBQ method starts from the observation that the above equation can be written as the sum of the squared errors, over each row of \\(W_{l}\\).


\\( \sum_{i=0}^{d_{row}} \|W_{l[i,:]}X-\hat{W}_{l[i,:]}X\|^{2}_{2} \\)

This means that we can quantize each row independently. This is called per-channel quantization. For each row \\(W_{l[i,:]}\\), OBQ quantizes one weight at a time while always updating all not-yet-quantized weights, in order to compensate for the error incurred by quantizing a single weight. The update on selected weights has a closed-form formula, utilizing Hessian matrices. 

The GPTQ paper improves this framework by introducing a set of optimizations that reduces the complexity of the quantization algorithm while retaining the accuracy of the model.

Compared to OBQ, the quantization step itself is also faster with GPTQ: it takes 2 GPU-hours to quantize a BERT model (336M) with OBQ, whereas with GPTQ, a Bloom model (176B) can be quantized in less than 4 GPU-hours. 

To learn more about the exact algorithm and the different benchmarks on perplexity and speedups, check out the original [paper](https://arxiv.org/pdf/2210.17323.pdf).

## AutoGPTQ library – the one-stop library for efficiently leveraging GPTQ for LLMs

The AutoGPTQ library enables users to quantize 🤗 Transformers models using the GPTQ method. While parallel community efforts such as [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa), [Exllama](https://github.com/turboderp/exllama) and [llama.cpp](https://github.com/ggerganov/llama.cpp/) implement quantization methods strictly for the Llama architecture, AutoGPTQ gained popularity through its smooth coverage of a wide range of transformer architectures.

Since the AutoGPTQ library has a larger coverage of transformers models, we decided to provide an integrated 🤗 Transformers API to make LLM quantization more accessible to everyone. At this time we have integrated the most common optimization options, such as CUDA kernels. For more advanced options like Triton kernels or fused-attention compatibility, check out the [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) library.

## Native support of GPTQ models in 🤗 Transformers

After [installing the AutoGPTQ library](https://github.com/PanQiWei/AutoGPTQ#quick-installation) and `optimum` (`pip install optimum`), running GPTQ models in Transformers is now as simple as:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", torch_dtype=torch.float16, device_map="auto")
```

Check out the Transformers [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) to learn more about all the features. 

Our AutoGPTQ integration has many advantages:

- Quantized models are serializable and can be shared on the Hub.
- GPTQ drastically reduces the memory requirements to run LLMs, while the inference latency is on par with FP16 inference.
- AutoGPTQ supports Exllama kernels for a wide range of architectures.
- The integration comes with native RoCm support for AMD GPUs.
- [Finetuning with PEFT](#--fine-tune-quantized-models-with-peft--) is available.

You can check on the Hub if your favorite model has already been quantized. TheBloke, one of Hugging Face top contributors, has quantized a lot of models with AutoGPTQ and shared them on the Hugging Face Hub. We worked together to make sure that these repositories will work out of the box with our integration.

This is a benchmark sample for the batch size = 1 case. The benchmark was run on a single NVIDIA A100-SXM4-80GB GPU. We used a prompt length of 512, and generated exactly 512 new tokens. The first row is the unquantized `fp16` baseline, while the other rows show memory consumption and performance using different AutoGPTQ kernels.

| gptq  | act_order | bits | group_size | kernel            | Load time (s) | Per-token latency (ms) | Throughput (tokens/s) | Peak memory (MB) |
|-------|-----------|------|------------|-------------------|---------------|------------------------|-----------------------|------------------|
| False | None      | None | None       | None              | 26.0          | 36.958                 | 27.058                | 29152.98         |
| True  | False     | 4    | 128        | exllama           | 36.2          | 33.711                 | 29.663                | 10484.34         |
| True  | False     | 4    | 128        | autogptq-cuda-old | 36.2          | 46.44                  | 21.53                 | 10344.62         |

A more comprehensive reproducible benchmark is available [here](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark).


## Quantizing models **with the Optimum library**

To seamlessly integrate AutoGPTQ into Transformers, we used a minimalist version of the AutoGPTQ API that is available in [Optimum](https://github.com/huggingface/optimum), Hugging Face's toolkit for training and inference optimization. By following this approach, we achieved easy integration with Transformers, while allowing people to use the Optimum API if they want to quantize their own models! Check out the Optimum [documentation](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization) if you want to quantize your own LLMs. 

Quantizing 🤗 Transformers models with the GPTQ method can be done in a few lines:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
```

Quantizing a model may take a long time. Note that for a 175B model, at least 4 GPU-hours are required if one uses a large dataset (e.g. `"c4"``). As mentioned above, many GPTQ models are already available on the Hugging Face Hub, which bypasses the need to quantize a model yourself in most use cases. Nevertheless, you can also quantize a model using your own dataset appropriate for the particular domain you are working on.

## Running GPTQ models through ***Text-Generation-Inference***

In parallel to the integration of GPTQ in Transformers, GPTQ support was added to the [Text-Generation-Inference library](https://github.com/huggingface/text-generation-inference) (TGI), aimed at serving large language models in production. GPTQ can now be used alongside features such as dynamic batching, paged attention and flash attention for a [wide range of architectures](https://huggingface.co/docs/text-generation-inference/main/en/supported_models).

As an example, this integration allows to serve a 70B model on a single A100-80GB GPU! This is not possible using a fp16 checkpoint as it exceeds the available GPU memory.

You can find out more about the usage of GPTQ in TGI in [the documentation](https://huggingface.co/docs/text-generation-inference/main/en/basic_tutorials/preparing_model#quantization).

Note that the kernel integrated in TGI does not scale very well with larger batch sizes. Although this approach saves memory, slowdowns are expected at larger batch sizes.

## **Fine-tune quantized models with PEFT**

You can not further train a quantized model using the regular methods. However, by leveraging the PEFT library, you can train adapters on top! To do that, we freeze all the layers of the quantized model and add the trainable adapters. Here are some examples on how to use PEFT with a GPTQ model: [colab notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing) and [finetuning](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) script. 

## Room for improvement

Our AutoGPTQ integration already brings impressive benefits at a small cost in the quality of prediction. There is still room for improvement, both in the quantization techniques and the kernel implementations.

First, while AutoGPTQ integrates (to the best of our knowledge) with the most performant W4A16 kernel (weights as int4, activations as fp16) from the [exllama implementation](https://github.com/turboderp/exllama), there is a good chance that the kernel can still be improved. There have been other promising implementations [from Kim et al.](https://arxiv.org/pdf/2211.10017.pdf) and from [MIT Han Lab](https://github.com/mit-han-lab/llm-awq) that appear to be promising. Moreover, from internal benchmarks, there appears to still be no open-source performant W4A16 kernel written in Triton, which could be a direction to explore.

On the quantization side, let’s emphasize again that this method only quantizes the weights. There have been other approaches proposed for LLM quantization that can quantize both weights and activations at a small cost in prediction quality, such as [LLM-QAT](https://arxiv.org/pdf/2305.17888.pdf) where a mixed int4/int8 scheme can be used, as well as quantization of the key-value cache. One of the strong advantages of this technique is the ability to use actual integer arithmetic for the compute, with e.g. [Nvidia Tensor Cores supporting int8 compute](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf). However, to the best of our knowledge, there are no open-source W4A8 quantization kernels available, but this may well be [an interesting direction to explore](https://www.qualcomm.com/news/onq/2023/04/floating-point-arithmetic-for-ai-inference-hit-or-miss).

On the kernel side as well, designing performant W4A16 kernels for larger batch sizes remains an open challenge.

### Supported models

In this initial implementation, only large language models with a decoder or encoder only architecture are supported. This may sound a bit restrictive, but it encompasses most state of the art LLMs such as Llama, OPT, GPT-Neo, GPT-NeoX.

Very large vision, audio, and multi-modal models are currently not supported.

## Conclusion and final words

In this blogpost we have presented the integration of the [AutoGPTQ library](https://github.com/PanQiWei/AutoGPTQ) in Transformers, making it possible to quantize LLMs with the GPTQ method to make them more accessible for anyone in the community and empower them to build exciting tools and applications with LLMs. 

This integration is available both for Nvidia GPUs, and RoCm-powered AMD GPUs, which is a huge step towards democratizing quantized models for broader GPU architectures.

The collaboration with the AutoGPTQ team has been very fruitful, and we are very grateful for their support and their work on this library.

We hope that this integration will make it easier for everyone to use LLMs in their applications, and we are looking forward to seeing what you will build with it!

Do not miss the useful resources shared above for better understanding the integration and how to quickly get started with GPTQ quantization.

- [Original Paper](https://arxiv.org/pdf/2210.17323.pdf)
- [Basic usage Google Colab notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) -  This notebook shows how to quantize your transformers model with GPTQ method, how to do inference, and how to do fine-tuning with the quantized model.
- Transformers integration [documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization)
- Optimum integration [documentation](https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization)
- The Bloke [repositories](https://huggingface.co/TheBloke?sort_models=likes#models) with compatible GPTQ models.


## Acknowledgements

We would like to thank [William](https://github.com/PanQiWei) for his support and his work on the amazing AutoGPTQ library and for his help in the integration. 
We would also like to thank [TheBloke](https://huggingface.co/TheBloke) for his work on quantizing many models with AutoGPTQ and sharing them on the Hub and for his help with the integration. 
We would also like to aknowledge [qwopqwop200](https://github.com/qwopqwop200) for his continuous contributions on AutoGPTQ library and his work on extending the library for CPU that is going to be released in the next versions of AutoGPTQ. 

Finally, we would like to thank [Pedro Cuenca](https://github.com/pcuenca) for his help with the writing of this blogpost.
