---
title: "Get your VLM running in 3 simple steps"
thumbnail: /blog/assets/optimum_intel/intel_thumbnail.png
authors:
- user: ezelanza
  guest: true
  org: Intel
- user: helenai
  guest: true
  org: Intel
- user: nikita-savelyev-intel
  guest: true
  org: Intel
- user: echarlaix
- user: IlyasMoutawwakil
---

# Get your VLM running in 3 simple steps on Intel CPUs

Deploy your [Vision Language Model (VLM)](https://huggingface.co/blog/vlms-2025) locally without the cost of cloud computing. This guide shows you how to use [Optimum Intel](https://huggingface.co/docs/optimum-intel/en/index) and [OpenVINO](https://docs.openvino.ai/2025/index.html) to get high-speed performance on your Intel CPU.

With the growing capability of large language models (LLMs), a new class of models has emerged: Vision Language Models (VLMs). These models can analyze images and videos to describe scenes, create captions, and answer questions about visual content.

While running AI models on your own device can be difficult as these models are often computationally demanding, it also offers significant benefits: including improved privacy since your data stays on your machine, and enhanced speed and reliability because you're not dependent on an internet connection or external servers. This is where tools like Optimum and OpenVINO come in, along with a small, efficient model like [SmolVLM](https://huggingface.co/blog/smolvlm). In this blog post, we'll walk you through three easy steps to get a VLM running locally, with no expensive hardware or GPUs required (though you can run all the code samples from this blog post on Intel GPUs).


## Deploy your model with Optimum

Small models like SmolVLM are built for low-resource consumption, but they can be further optimized. In this blog post we will see how to optimize your model, to lower memory usage and speedup inference, making it more efficient for deployment on devices with limited resources.

To follow this tutorial, you need to install `optimum` and `openvino`, which you can do with:

```bash
pip install optimum-intel[openvino]
```

## Step 1: Convert your model

First, you will need to convert your model to the [OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html). There are multiple options to do it:

1. You can use the [Optimum CLI](https://huggingface.co/docs/optimum-intel/en/openvino/export#using-the-cli)

```bash
optimum-cli export openvino -m HuggingFaceTB/SmolVLM2-256M-Video-Instruct smolvlm_ov/
```

2. Or you can convert it [on the fly](https://huggingface.co/docs/optimum-intel/en/openvino/export#when-loading-your-model) when loading your model:


```python
from optimum.intel import OVModelForVisualCausalLM

model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
model = OVModelForVisualCausalLM.from_pretrained(model_id)
model.save_pretrained("smolvlm_ov")
```

## Step 2: Quantization

Now it’s time to optimize your model. Quantization reduces the precision of the model weights and/or activations, leading to smaller, faster models.

Essentially, it's a way to map values from a high-precision data type, such as 32-bit floating-point numbers (FP32), to a lower-precision format, typically 8-bit integers (INT8). While this process offers several key benefits, it can also impact in a potential loss of accuracy.

<figure style="width: 800px; margin: 0 auto;">
  <img src="https://huggingface.co/datasets/openvino/documentation/resolve/main/blog/openvino_vlm/quantization.png">
</figure>

Optimum supports two main post-training quantization methods:

- Weight Only Quantization
- Static Quantization

Let’s explore each of them.

### Option 1: Weight Only Quantization

Weight-only quantization means that only the weights are quantized but activations remain in their original precisions. As a result, the model becomes smaller and more memory-efficient, improving loading times. But since activations are not quantized, inference speed gains are limited. Since OpenVINO 2024.3, if the model's weight have been quantized, the corresponding activations will also be quantized at runtime, leading to additional speedup depending on the device.

Weight-only quantization is a simple first step since it usually doesn’t result in significant accuracy degradation.  
In order to run it, you will need to create a quantization configuration `OVWeightQuantizationConfig` as follows:

```python
from optimum.intel import OVModelForVisualCausalLM, OVWeightQuantizationConfig

q_config = OVWeightQuantizationConfig(bits=8)
# Apply quantization and save the new model
q_model = OVModelForVisualCausalLM.from_pretrained(model_id, quantization_config=q_config)
q_model.save_pretrained("smolvlm_int8")
```

or equivalently using the CLI:

```bash
optimum-cli export openvino -m HuggingFaceTB/SmolVLM2-256M-Video-Instruct --weight-format int8 smolvlm_int8/
```

## Option 2: Static Quantization

With Static Quantization, both weights and activations are quantized before inference. To achieve the best estimate for the activation quantization parameters, we perform a calibration step. During this step, a small representative dataset is fed through the model. In our case, we will use 50 samples of the [contextual dataset](https://huggingface.co/datasets/ucla-contextual/contextual_test).

```python
from optimum.intel import OVModelForVisualCausalLM, OVQuantizationConfig

q_config = OVQuantizationConfig(bits=8, dataset="contextual", num_samples=50)
q_model = OVModelForVisualCausalLM.from_pretrained(model_id, quantization_config=q_config)
q_model.save_pretrained("smolvlm_static_int8")
```

or equivalently with the CLI:

```bash
optimum-cli export openvino -m HuggingFaceTB/SmolVLM2-256M-Video-Instruct --quant-mode int8 --dataset contextual --num-samples 50 smolvlm_static_int8/
```

Quantizing activations adds small errors that can build up and affect accuracy, so careful testing afterward is important. More information and examples can be found in [our documentation](https://huggingface.co/docs/optimum-intel/en/openvino/optimization#pipeline-quantization).

### Step 3: Run inference

You can now run inference with your quantized model:

```python
generated_ids = q_model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

If you have a recent Intel laptop, Intel AI PC, or Intel discrete GPU, you can load the model on GPU by adding `device="gpu"` when loading your model:

```python
model = OVModelForVisualCausalLM.from_pretrained(model_id, device="gpu")
```

Try the complete notebook [here](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/vision_language_quantization.ipynb).

We also created a [space](https://huggingface.co/spaces/echarlaix/vision-langage-openvino) so you can play with the [original model](https://huggingface.co/echarlaix/SmolVLM2-500M-Video-Instruct-openvino) and its quantized variants obtained by respectively applying [weight-only quantization](https://huggingface.co/echarlaix/SmolVLM2-500M-Video-Instruct-openvino-8bit-woq) and [static quantization](https://huggingface.co/echarlaix/SmolVLM2-500M-Video-Instruct-openvino-8bit-static). This demo runs on 4th Generation Intel Xeon (Sapphire Rapids) processors.

<figure style="width: 700px; margin: 0 auto;">
  <img src="https://huggingface.co/datasets/openvino/documentation/resolve/main/blog/openvino_vlm/chat1.png">
</figure>

## Evaluation and Conclusion

Multimodal AI is becoming more accessible thanks to smaller, optimized models like SmolVLM and tools such as Hugging Face Optimum and OpenVINO. While deploying vision-language models locally still presents challenges, this workflow shows that it's possible to run lightweight image-and-text models on multiple hardware.

We ran a benchmark to compare the performance of the PyTorch, OpenVINO, and OpenVINO 8-bit weight-only quantized (WOQ) versions of the (SmolVLM2-256M)[https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct] model. The goal was to evaluate the impact of weight-only quantization on latency and throughput on Intel CPU hardware. For this test, we used a single image as input.

We measured the following metrics to evaluate the model's performance:

- Time To First Token (TTFT) : Time it takes to generate the first output token.
- Time Per Output Token (TPOT): Time it takes to generate each subsequent output tokens.
- End-to-End Latency : Total time it takes to generate the output all output tokens.
- Decoding Throughput: Number of tokens per second the model generates during the decoding phase.

<p align="center">
  <img src="https://huggingface.co/datasets/OpenVINO/documentation/resolve/main/blog/openvino_vlm/flower.png" alt="Pink flower with bee" width="700"/>
</p>


Here are the results on Intel CPU:

| Configuration    |        TTFT        |      TPOT        | End-to-End Latency    | Decoding Throughput           |
|------------------|--------------------|------------------|-----------------------|-------------------------------|
| pytorch          | 5.150              | 1.385            | 25.927                | 0.722                         |
| openvino         | 0.420              | 0.021            | 0.738                 | 47.237                        |
| openvino-8bit-woq| 0.247              | 0.016            | 0.482                 | 63.928                        |


This benchmark shows that small, optimized multimodal models, like (SmolVLM2-256M)[https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct], can run efficiently on various Intel hardware. Weight-only quantization significantly reduces model size, improving efficiency without majorly impacting throughput. GPUs deliver the highest image and token processing speeds, while CPUs and iGPUs remain viable for lighter workloads. Overall, this shows that lightweight vision-language models can be deployed locally with reasonable performance, making multimodal AI more accessible.

## Useful Links & Resources
- [Notebook](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/vision_language_quantization.ipynb)
- [Try our Space](https://huggingface.co/spaces/echarlaix/vision-langage-openvino)
- [Watch the webinar recording](https://web.cvent.com/event/d550a2a7-04f2-4a28-b641-3af228e318ca/regProcessStep1?utm_campaign=speakers4&utm_medium=organic&utm_source=Community)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum-intel/en/openvino/inference)
