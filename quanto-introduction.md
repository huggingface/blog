---
title: "Quanto: a PyTorch quantization backend for Optimum"
thumbnail: /blog/assets/169_quanto_intro/thumbnail.png
authors:
- user: dacorvo
- user: ybelkada
- user: marcsun13
---

# Quanto: a PyTorch quantization backend for Optimum

Quantization is a technique to reduce the computational and memory costs of evaluating Deep Learning Models by representing their weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).

Reducing the number of bits means the resulting model requires less memory storage, which is crucial for deploying Large Language Models on consumer devices.
It also enables specific optimizations for lower bitwidth datatypes, such as `int8` or `float8` matrix multiplications on CUDA devices.

Many open-source libraries are available to quantize pytorch Deep Learning Models, each providing very powerful features, yet often restricted to specific model configurations and devices.

Also, although they are based on the same design principles, they are unfortunately often incompatible with one another.

Today, we are excited to introduce [quanto](https://github.com/huggingface/optimum-quanto), a PyTorch quantization backend for [Optimum](https://huggingface.co/docs/optimum/index).

It has been designed with versatility and simplicity in mind:

- all features are available in eager mode (works with non-traceable models),
- quantized models can be placed on any device (including CUDA and MPS),
- automatically inserts quantization and dequantization stubs,
- automatically inserts quantized functional operations,
- automatically inserts quantized modules (see below the list of supported modules),
- provides a seamless workflow from a float model to a dynamic to a static quantized model,
- serialization compatible with PyTorch `weight_only` and 🤗 [Safetensors](https://huggingface.co/docs/safetensors/index),
- accelerated matrix multiplications on CUDA devices (int8-int8, fp16-int4, bf16-int8, bf16-int4),
- supports int2, int4, int8 and float8 weights,
- supports int8 and float8 activations.

Recent quantization methods appear to be focused on quantizing Large Language Models (LLMs), whereas [quanto](https://github.com/huggingface/optimum-quanto) intends to provide extremely simple quantization primitives for simple quantization schemes (linear quantization, per-group quantization) that are adaptable across any modality.

## Quantization workflow

Quanto is available as a pip package.

```sh
pip install optimum-quanto
```

A typical quantization workflow consists of the following steps:

**1. Quantize**

The first step converts a standard float model into a dynamically quantized model.

```python
from optimum.quanto import quantize, qint8

quantize(model, weights=qint8, activations=qint8)
```

At this stage, only the inference of the model is modified to dynamically quantize the weights.

**2. Calibrate (optional if activations are not quantized)**

Quanto supports a calibration mode that allows the recording of the activation ranges while passing representative samples through the quantized model.

```python
from optimum.quanto import Calibration

with Calibration(momentum=0.9):
    model(samples)
```

This automatically activates the quantization of the activations in the quantized modules.


**3. Tune, aka Quantization-Aware-Training (optional)**

If the performance of the model degrades too much, one can tune it for a few epochs to recover the float model performance.

```python
import torch

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data).dequantize()
    loss = torch.nn.functional.nll_loss(output, target)
    loss.backward()
    optimizer.step()
```

**4. Freeze integer weights**

When freezing a model, its float weights are replaced by quantized weights.

```python
from optimum.quanto import freeze

freeze(model)
```

**5. Serialize quantized model**

Quantized models weights can be serialized to a `state_dict`, and saved to a file.
Both `pickle` and `safetensors` (recommended) are supported.

```python
from safetensors.torch import save_file

save_file(model.state_dict(), 'model.safetensors')
```

In order to reload these weights, you also need to store the quantized
models quantization map.

```python
import json

from optimum.quanto import quantization_map

with open('quantization_map.json', w) as f:
  json.dump(quantization_map(model))
```

**5. Reload a quantized model**

A serialized quantized model can be reloaded from a `state_dict` and a `quantization_map` using the `requantize` helper.
Note that you need to first instantiate an empty model.

```python
import json

from safetensors.torch import load_file

state_dict = load_file('model.safetensors')
with open('quantization_map.json', r) as f:
  quantization_map = json.load(f)

# Create an empty model from your modeling code and requantize it
with torch.device('meta'):
  new_model = ...
requantize(new_model, state_dict, quantization_map, device=torch.device('cuda'))
```

Please refer to the [examples](https://github.com/huggingface/optimum-quanto/tree/main/examples) for instantiations of the quantization workflow.
You can also check this [notebook](https://colab.research.google.com/drive/1qB6yXt650WXBWqroyQIegB-yrWKkiwhl?usp=sharing) where we show you how to quantize a BLOOM model with quanto!
## Performance

Below are two graphs evaluating the accuracy of different quantized configurations for [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B.

Note: the first bar in each group always corresponds to the non-quantized model.

<div class="row"><center>
  <div class="column">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/quanto-introduction/meta-llama-Meta-Llama-3.1-8B_bf16_Perplexity.png" alt=""meta-llama/Meta-Llama-3.1-8B WikiText perplexity">
  </div>
 </center>
</div>

These results are obtained without applying any Post-Training-Optimization algorithm like [hqq](https://mobiusml.github.io/hqq_blog/) or [AWQ](https://github.com/mit-han-lab/llm-awq).

The graph below gives the latency per-token measured on an NVIDIA A10 GPU.

<div class="row"><center>
  <div class="column">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/quanto-introduction/meta-llama-Meta-Llama-3.1-8B_bf16_Latency__ms_.png" alt="meta-llama/Meta-Llama-3.1-8B Mean latency per token">
  </div>
 </center>
</div>

Stay tuned for updated results as we are constantly improving [quanto](https://github.com/huggingface/optimum-quanto) with optimizers and optimized kernels.

Please refer to the [quanto benchmarks](https://github.com/huggingface/optimum-quanto/tree/main/bench/) for detailed results for different model architectures and configurations.

## Integration in transformers


Quanto is seamlessly integrated in the Hugging Face [transformers](https://github.com/huggingface/transformers) library. You can quantize any model by passing a `QuantoConfig` to `from_pretrained`!

Currently, you need to use the latest version of [accelerate](https://github.com/huggingface/accelerate) to make sure the integration is fully compatible.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = QuantoConfig(weights="int8")

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config= quantization_config
)
```

You can quantize the weights and/or activations in int8, float8, int4, or int2 by simply passing the correct argument in `QuantoConfig`. The activations can be either in int8 or float8. For float8, you need to have hardware that is compatible with float8 precision, otherwise quanto will silently upcast the weights and activations to torch.float32 or torch.float16 (depending on the original data type of the model) when we perform the matmul (only when the weight is quantized). If you try to use `float8` using MPS devices, `torch` will currently raise an error.

Quanto is device agnostic, meaning you can quantize and run your model regardless if you are on CPU/GPU/ MPS (Apple Silicon).

Quanto is also torch.compile friendly. You can quantize a model with quanto and call `torch.compile` to the model to compile it for faster generation. This feature might not work out of the box if dynamic quantization is involved (i.e., Quantization Aware Training or quantized activations enabled). Make sure to keep `activations=None` when creating your `QuantoConfig` in case you use the transformers integration.

It is also possible to quantize any model, regardless of the modality using quanto! We demonstrate how to quantize `openai/whisper-large-v3` model in int8 using quanto.

```python
from transformers import AutoModelForSpeechSeq2Seq

model_id = "openai/whisper-large-v3"
quanto_config = QuantoConfig(weights="int8")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
   model_id,
   torch_dtype=torch.float16,
   device_map="cuda",
   quantization_config=quanto_config
)
```

Check out this [notebook](https://colab.research.google.com/drive/16CXfVmtdQvciSh9BopZUDYcmXCDpvgrT?usp=sharing#scrollTo=IHbdLXAg53JL) for a complete tutorial on how to properly use quanto with the transformers integration!

## Contributing to quanto

Contributions to [quanto](https://github.com/huggingface/optimum-quanto) are very much welcomed, especially in the following areas:

- optimized kernels for [quanto](https://github.com/huggingface/optimum-quanto) operations targeting specific devices,
- Post-Training-Quantization optimizers to recover the accuracy lost during quantization,
- helper classes for `transformers` or `diffusers` models.
