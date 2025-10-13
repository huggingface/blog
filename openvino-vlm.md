---
title: "Get your VLM running in 3 simple steps on Intel CPUs"
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

With the growing capability of large language models (LLMs), a new class of models has emerged: [Vision Language Models (VLMs)](https://huggingface.co/blog/vlms-2025). These models can analyze images and videos to describe scenes, create captions, and answer questions about visual content.

While running AI models on your own device can be difficult as these models are often computationally demanding, it also offers significant benefits: including improved privacy since your data stays on your machine, and enhanced speed and reliability because you're not dependent on an internet connection or external servers. This is where tools like [Optimum Intel](https://huggingface.co/docs/optimum-intel/en/index) and [OpenVINO](https://docs.openvino.ai/2025/index.html) come in, along with a small, efficient model like [SmolVLM](https://huggingface.co/blog/smolvlm). In this blog post, we'll walk you through three easy steps to get a VLM running locally, with no expensive hardware or GPUs required (though you can run all the code samples from this blog post on Intel GPUs).


## Deploy your model with Optimum

Small models like SmolVLM are built for low-resource consumption, but they can be further optimized. In this blog post we will see how to optimize your model, to lower memory usage and speedup inference, making it more efficient for deployment on devices with limited resources.

To follow this tutorial, you need to install `optimum` and `openvino`, which you can do with:

```bash
pip install optimum-intel[openvino] transformers==4.52.*
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

Now it’s time to optimize your model. Quantization reduces the precision of the model weights and/or activations, leading to smaller, faster models. Essentially, it's a way to map values from a high-precision data type, such as 32-bit floating-point numbers (FP32), to a lower-precision format, typically 8-bit integers (INT8). While this process offers several key benefits, it can also impact in a potential loss of accuracy.

<p align="center">
  <img src="https://huggingface.co/datasets/OpenVINO/documentation/resolve/main/blog/openvino_vlm/quantization.png" alt="Quantization" width="700"/>
</p>

Optimum supports two main post-training quantization methods:

- [Weight Only Quantization (WOQ)](https://huggingface.co/docs/optimum-intel/en/openvino/optimization#weight-only-quantization)
- [Static Quantization](https://huggingface.co/docs/optimum-intel/en/openvino/optimization#full-quantization)

Let’s explore each of them.

### Option 1: Weight Only Quantization

Weight-only quantization means that only the weights are quantized but activations remain in their original precisions. As a result, the model becomes smaller and more memory-efficient, improving loading times. But since activations are not quantized, inference speed gains are limited. Weight-only quantization is a simple first step since it usually doesn’t result in significant accuracy degradation.

> [!NOTE]
> Since OpenVINO 2024.3, if the model's weight have been quantized, the corresponding activations will also be quantized at runtime, leading to additional speedup depending on the device.

In order to run it, you will need to create a quantization configuration `OVWeightQuantizationConfig` as follows:

```python
from optimum.intel import OVModelForVisualCausalLM, OVWeightQuantizationConfig

q_config = OVWeightQuantizationConfig(bits=8)
q_model = OVModelForVisualCausalLM.from_pretrained(model_id, quantization_config=q_config)
q_model.save_pretrained("smolvlm_int8")
```

or equivalently using the CLI:

```bash
optimum-cli export openvino -m HuggingFaceTB/SmolVLM2-256M-Video-Instruct --weight-format int8 smolvlm_int8/
```

## Option 2: Static Quantization

With Static Quantization, both weights and activations are quantized before inference. To achieve the best estimate for the activation quantization parameters, we perform a calibration step. During this step, a small representative dataset is fed through the model. In our case, we will use 50 samples of the [contextual dataset](https://huggingface.co/datasets/ucla-contextual/contextual_test) and will apply static quantization on the vision encoder while weight-only quantization will be applied on the rest of the model. Experiments show that applying static quantization on the vision encoder provides a noticeable performance improvement without significant accuracy degradation. Since the vision encoder is called only once per generation, the overall performance gain from applying static quantization on this component is lower than the gain achieved by optimizing more frequently used components like the language model. Nevertheless, this approach can be beneficial in certain scenarios. For example, when short answers are needed, especially with multiple images as input.

```python
from optimum.intel import OVModelForVisualCausalLM, OVPipelineQuantizationConfig, OVQuantizationConfig, OVWeightQuantizationConfig

q_config = OVPipelineQuantizationConfig(
    quantization_configs={
        "lm_model": OVWeightQuantizationConfig(bits=8),
        "text_embeddings_model": OVWeightQuantizationConfig(bits=8),
        "vision_embeddings_model": OVQuantizationConfig(bits=8),
    },
    dataset=dataset,
    num_samples=num_samples,
)
q_model = OVModelForVisualCausalLM.from_pretrained(model_id, quantization_config=q_config)
q_model.save_pretrained("smolvlm_static_int8")
```

Quantizing activations adds small errors that can build up and affect accuracy, so careful testing afterward is important. More information and examples can be found in [our documentation](https://huggingface.co/docs/optimum-intel/en/openvino/optimization#pipeline-quantization).

### Step 3: Run inference

You can now run inference with your quantized model:

```python
generated_ids = q_model.generate(**inputs, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts[0])
```

If you have a recent Intel laptop, Intel AI PC, or Intel discrete GPU, you can load the model on GPU by adding `device="gpu"` when loading your model:

```python
model = OVModelForVisualCausalLM.from_pretrained(model_id, device="gpu")
```

We also created a [space](https://huggingface.co/spaces/echarlaix/vision-langage-openvino) so you can play with the [original model](https://huggingface.co/echarlaix/SmolVLM2-256M-Video-Instruct-openvino) and its quantized variants obtained by respectively applying [weight-only quantization](https://huggingface.co/echarlaix/SmolVLM2-256M-Video-Instruct-openvino-8bit-woq-data-free) and [mixed quantization](https://huggingface.co/echarlaix/SmolVLM2-256M-Video-Instruct-openvino-8bit-mixed). This demo runs on 4th Generation Intel Xeon (Sapphire Rapids) processors.


<p align="center">
  <img src="https://huggingface.co/datasets/OpenVINO/documentation/resolve/main/blog/openvino_vlm/chat1.png" alt=" HF Space" width="500"/>
</p>

To reproduce our results, check out our [notebook](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/vision_language_quantization.ipynb).

## Evaluation and Conclusion

We ran a benchmark to compare the performance of the [PyTorch](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct), [OpenVINO](https://huggingface.co/echarlaix/SmolVLM2-256M-Video-Instruct-openvino), and [OpenVINO 8-bit WOQ](https://huggingface.co/echarlaix/SmolVLM2-256M-Video-Instruct-openvino-8bit-woq-data-free) versions of the original model. The goal was to evaluate the impact of weight-only quantization on latency and throughput on Intel CPU hardware. For this test, we used [a single image](https://huggingface.co/datasets/OpenVINO/documentation/resolve/main/blog/openvino_vlm/flower.png) as input.

We measured the following metrics to evaluate the model's performance:
- Time To First Token (TTFT) : Time it takes to generate the first output token.
- Time Per Output Token (TPOT): Time it takes to generate each subsequent output tokens.
- End-to-End Latency : Total time it takes to generate the output all output tokens.
- Decoding Throughput: Number of tokens per second the model generates during the decoding phase.

Here are the results on Intel CPU:

| Configuration    |Time To First Token (TTFT)|Time Per Output Token (TPOT)| End-to-End Latency    | Decoding Throughput           |
|------------------|--------------------------|----------------------------|-----------------------|-------------------------------|
| pytorch          | 5.150                    | 1.385                      | 25.927                | 0.722                         |
| openvino         | 0.420                    | 0.021                      | 0.738                 | 47.237                        |
| openvino-8bit-woq| 0.247                    | 0.016                      | 0.482                 | 63.928                        |


This benchmark demonstrates how small, optimized multimodal models, like [SmolVLM2-256M](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct), perform on Intel CPUs across different configurations. According to the tests, the PyTorch version shows high latency, with a time to first token (TTFT) of over 5s with a decoding throughput of ~0.7 tokens/s. Simply converting the model with Optimum and running it on OpenVINO drastically reduces the time to first token (TTFT) to 0.42s (~x12 speedup) and eaises throughput to ~47 tokens/s (~x65). Applying 8-bit weight-only quantization further reduces TTFT (x1.7) and increases throughput (x1.4), while also reducing model size and improving efficiency.

> [!NOTE]
> **Platform configuration**
> Platform Configuration for performance claims above:
>
> **System Board:** MSI B860M GAMING PLUS WIFI (MS-7E42)  
> **CPU:** Intel® Core™ Ultra 7 265K  
> **Sockets/Physical Cores:** 1/20 (20 threads)  
> **HyperThreading/Turbo Settings:** Disabled  
> **Memory:** 64 GB DDR5 @ 6400 MHz  
> **GPU:** 1x Intel® Graphics (4 Xe cores), 1x Intel® Arc™ A770 Graphics (32 Xe cores), 1x Intel® Arc™ B580 Graphics (20 Xe cores)  
> **NPU:** Intel® AI Boost  
> **TDP:** 665W  
> **BIOS:** American Megatrends International, LLC. 2.A10  
> **BIOS Release Date:** 28.11.2024  
> **OS:** Ubuntu 24.10  
> **Kernel:** 6.11.0–25-generic  
> **OpenVINO Version:** 2025.2.0  
> **torch:** 2.8.0  
> **torchvision:** 0.23.0+cpu  
> **optimum-intel:** 1.25.2  
> **transformers:** 4.53.3  
> **Benchmark Date:** 15.05.2025  
> **Benchmarked by:** Intel Corporation
> Performance may vary by use, configuration, and other factors. See the platform configuration below.


## Useful Links & Resources

- [Notebook](https://github.com/huggingface/optimum-intel/blob/main/notebooks/openvino/vision_language_quantization.ipynb)
- [Try our Space](https://huggingface.co/spaces/echarlaix/vision-langage-openvino)
- [Watch the webinar recording](https://web.cvent.com/event/d550a2a7-04f2-4a28-b641-3af228e318ca/regProcessStep1?utm_campaign=speakers4&utm_medium=organic&utm_source=Community)
- [Optimum Intel Documentation](https://huggingface.co/docs/optimum-intel/en/openvino/inference)


> [!NOTE]
> ## Notices & Disclaimers
> Performance varies by use, configuration, and other factors. Learn more on the Performance Index site.
> Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates. See backup for configuration details. No product or component can be absolutely secure. Your costs and results may vary. Intel technologies may require enabled hardware, software or service activation.
> © Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries.