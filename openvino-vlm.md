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

Early models like [Flamingo](https://arxiv.org/abs/2204.14198) and [Idefics](https://huggingface.co/blog/idefics) showed what was possible. Both demonstrated interesting capabilities, using 80B parameters. More recently, we’ve seen much smaller models emerge, like [PaliGemma 2 (3B)](https://huggingface.co/google/paligemma2-3b-pt-224), [moondream2 (2B)](https://huggingface.co/vikhyatk/moondream2), or [Qwen2.5-VL (7B)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), but even these “small” versions can be tough to run locally because they still carry a lot of the memory and compute demands from their larger predecessors.

While running AI models on your own device can be difficult as these models are often computationally demanding, it also offers significant benefits: including improved privacy since your data stays on your machine, and enhanced speed and reliability because you're not dependent on an internet connection or external servers. This is where tools like Optimum and OpenVINO come in, along with a small, efficient model like [SmolVLM](https://huggingface.co/blog/smolvlm). In this blog post, we'll walk you through three easy steps to get a VLM running locally, with no expensive hardware or GPUs required (though you can run all the code samples from this blog post on Intel GPUs).

## Optimum

Even though SmolVLM was designed for low-resource consumption, there’s still room for improvement. These models can be further compressed or optimized for your own hardware. However, if you’ve tried to optimize a model yourself, you probably know it’s not a trivial task.

Before using it, the very first step is to install the library.  
```bash  
pip install optimum-intel[openvino]  
```

By using Optimum with OpenVINO, you gain several benefits, like improving the inference time and lower memory/storage usage out of the box. But you can go even further: quantization can reduce the model size and resource consumption even more. While quantization often requires deep expertise, Optimum simplifies the process, making it much more accessible.

Let’s see how you can run SmolVLM then.

## Step 1: Convert your model to the OpenVINO IR

First, you will need to convert your model to the OpenVINO IR. There are multiple options to do it:

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

Now it’s time to optimize the model for efficient execution using **quantization**. Quantization reduces the precision of the model weights and/or activations, leading to smaller, faster models.

Essentially, it's a way to map values from a high-precision data type, such as 32-bit floating-point numbers (FP32), to a lower-precision format, typically 8-bit integers (INT8). While this process offers several key benefits, it can also impact in a potential loss of accuracy.

<figure style="width: 800px; margin: 0 auto;">
  <img src="https://huggingface.co/datasets/openvino/documentation/resolve/main/blog/openvino_vlm/quantization.png">
</figure>

Optimum supports two main post-training quantization methods:

- Weight Only Quantization  
- Static Quantization

Let’s explore each of them.

### Option 1: Weight Only Quantization

Weight-only quantization means that only the weights are quantized but activations remain in their original precisions. To explain this process, let’s imagine preparing for a long backpacking trip. To reduce weight, you replace bulky items like full-size shampoo bottles with compact travel-sized versions. This is like weight-only quantization, where the model’s weights are compressed from 32-bit floating-point numbers to 8-bit integers, reducing the model’s memory footprint.

However, the “interactions” during the trip, like drinking water, remain unchanged. This is similar to what happens to activations, which stay in high precision (FP32 or BF16) to preserve accuracy during computation.

As a result, the model becomes smaller and more memory-efficient, improving loading times. But since activations are not quantized, inference speed gains are limited. Since OpenVINO 2024.3, if the model's weight have been quantized, the corresponding activations will also be quantized at runtime, leading to additional speedup depending on the device.

Weight-only quantization is a simple first step since it usually doesn’t result in significant accuracy degradation.  
In order to run it, you will need to create a quantization configuration using Optimum `OVWeightQuantizationConfig` as follows


```python
from optimum.intel import OVModelForVisualCausalLM, OVWeightQuantizationConfig

q_config = OVWeightQuantizationConfig(bits=8)
# Apply quantization and save the new model
q_model = OVModelForVisualCausalLM.from_pretrained(model_id, quantization_config=q_config)
q_model.save_pretrained("smolvlm_int8")
```

or quivalently using the CLI:


```bash
optimum-cli export openvino -m HuggingFaceTB/SmolVLM2-256M-Video-Instruct --weight-format int8 smolvlm_int8/

```

## Option 2: Static Quantization

To achieve the best estimate for the activation quantization parameters, we perform a calibration step. This involves running inference on a small subset of our dataset, in our case using 50 samples of the [contextual dataset](https://huggingface.co/datasets/ucla-contextual/contextual_test).

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

## Evaluation and Conclusion

Multimodal AI is becoming more accessible thanks to smaller, optimized models like SmolVLM and tools such as Hugging Face Optimum and OpenVINO. While deploying vision-language models locally still presents challenges, this workflow shows that it's possible to run lightweight image-and-text models on multiple hardware.

We ran a benchmark to show the impact of weight-only quantization on a (SmolVLM2-256M)[https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct] model and how it performs on different Intel hardware. For this test, we used a single image.
We measured the following metrics to evaluate the model's performance:
- Model Size: this shows how much storage space the model requires.
- Latency: we measured the average time it took for the model to process an input.
- Image throughput: the rate at which the model can process images.
- Tokens throughput: the rate at which the model can process tokens.

<p align="center">
  <img src="https://huggingface.co/datasets/OpenVINO/documentation/resolve/main/blog/openvino_vlm/flower.png" alt="Pink flower with bee" width="700"/>
</p>


Here are the results across different Intel hardware:

| Device       | Model Size (MB) (Before/After) | Images Throughput (im/s) (Before/After) | First Token Throughput (t/s) (Before/After) | Second Token Throughput (t/s) (Before/After) | Latency (s) (Before/After) |
|-------------|-------------------------------|-----------------------------------------|--------------------------------------------|---------------------------------------------|-----------------------------|
| CPU         |  -             | 0.33 / 0.55                              | 2.69 / 3.94                                | 83.25 / 146.1                               | 3.5249 / 2.1548            |
| iGPU        | -                             | 0.58 / 0.53                              | 5.01 / 5.26                                | 51.62 / 49.56                               | 2.1386 / 2.3182            |
| GPU (b580)  | 980.61 / 248 (Applies to all devices)                             | 15.75 / 15.01                            | 34.51 / 27.54                              | 149.79 / 120.91                             | 0.2074 / 0.2376            |
| GPU (A770)  | -                             | 10.68 / 10.89                            | 16.57 / 15.79                              | 83.01 / 69.1                                | 0.3321 / 0.3403            |

This benchmark shows that small, optimized multimodal models, like (SmolVLM2-256M)[https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct], can run efficiently on various Intel hardware. Weight-only quantization significantly reduces model size, improving efficiency without majorly impacting throughput. GPUs deliver the highest image and token processing speeds, while CPUs and iGPUs remain viable for lighter workloads. Overall, this shows that lightweight vision-language models can be deployed locally with reasonable performance, making multimodal AI more accessible.


## Useful Links & Resources
- [Notebook](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/vision_language_quantization.ipynb)
- [Try our Space](https://huggingface.co/spaces/echarlaix/vision-langage-openvino)
- [Watch the webinar recording](https://web.cvent.com/event/d550a2a7-04f2-4a28-b641-3af228e318ca/regProcessStep1?utm_campaign=speakers4&utm_medium=organic&utm_source=Community)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum-intel/en/openvino/inference)
