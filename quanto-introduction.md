---
title: "Quanto: a pytorch quantization toolkit"
thumbnail: /blog/assets/quanto-intro/thumbnail.png
authors:
- user: dacorvo
- user: ybelkada
- user: marcsun13
---

# Quanto: a pytorch quantization toolkit

Quantization is a technique to reduce the computational and memory costs of evaluating Deep Learning Models by representing their weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).

Reducing the number of bits means the resulting model requires less memory storage, which is crucial for deploying Large Language Models on consumer devices.
It also allows to take advantage of specific optimizations for lower bitwidth datatypes, such as `int8` of `float8` matrix multiplications on CUDA devices.

Many open-source libraries are available to quantize pytorch Deep Learning Models, each providing very powerful features, yet often restricted to specific model configurations and devices.

Also, although being based on the same design principles, they are unfortunately most of the time incompatible with one another.

Today, we are excited to introduce [🤗 quanto](https://github.com/huggingface/quanto), a versatile pytorch quantization toolkit, that provides several unique features:

- available in eager mode (works with non-traceable models),
- quantized models can be placed on any device (including CUDA and MPS),
- automatically inserts quantization and dequantization stubs,
- automatically inserts quantized functional operations,
- automatically inserts quantized modules (see below the list of supported modules),
- provides a seamless workflow from a float model to a dynamic to a static quantized model,
- supports quantized model serialization as a `state_dict`,
- supports not only `int8` weights, but also `int2` and `int4`,
- supports not only `int8` activations, but also `float8`.

Recent quantization methods appear to be focussed on quantizing Large Language Models (LLMs), whereas quanto intends to provide extremely simple quantization primitives for using simple quantization schemes (linear quantization, per-group quantization) that are adaptable across any modality.

The goal of [🤗 quanto](https://github.com/huggingface/quanto) is not to replace other quantization libraries, but to foster innovation by lowering the bar
to implement and combine quantization features.

Make no mistake, quantization is hard, and integrating it seamlessly in existing models requires a deep understanding of pytorch internals.
But don't worry: [🤗 quanto](https://github.com/huggingface/quanto)'s goal is to do most of the heavy-lifting for you, so that you can focus
on what matters most: exploring the realms of low-bitwidth machine learning and finding solutions for the GPU poor.

## Quantization workflow

Quanto is available as a pip package.

```sh
pip install quanto
```

[🤗 quanto](https://github.com/huggingface/quanto) does not make a clear distinction between dynamic and static quantization: models are first dynamically quantized,
but their weights can later be "frozen" to static values.

A typical quantization workflow consists of the following steps:

**1. Quantize**

The first step converts a standard float model into a dynamically quantized model.

```python
quantize(model, weights=quanto.qint8, activations=quanto.qint8)
```

At this stage, only the inference of the model is modified to dynamically quantize the float weights.

**2. Calibrate (optional if activations are not quantized)**

Quanto supports a calibration mode that allows to record the activation ranges while passing representative samples through the quantized model.

```python
with calibration(momentum=0.9):
    model(samples)
```

This automatically activates the quantization of the activations in the quantized modules.

**3. Tune, aka Quantization-Aware-Training (optional)**

If the performance of the model degrades too much, one can tune it for a few epochs to try to recover the float model performance.

```python
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

When freezing a model, its float weights are replaced by quantized integer weights.

```python
freeze(model)
```

Please refer to the [examples](https://github.com/huggingface/quanto/tree/main/examples) for instantiations of that workflow.

## Performances

These are some very preliminary results, as we are constantly improving both the accuracy and efficiency of quantized models, but it looks already very promising.

Below are two graphs evaluating the accuracy of different quantized configurations for [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).

Note: the first bar in each group always corresponds to the non-quantized model.

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-v0.1_Accuracy.png" alt="mistralai/Mistral-7B-v0.1 Lambada prediction accuracy">
  </div>
 </center>
</div>

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-v0.1_Perplexity.png" alt="mistralai/Mistral-7B-v0.1 Lambada prediction accuracy">
  </div>
 </center>
</div>

The graph below gives the latency per-token measured on an NVIDIA A100 GPU.

<div class="row"><center>
  <div class="column">
    <img src="https://github.com/huggingface/quanto/blob/main/bench/generation/charts/mistralai-Mistral-7B-v0.1_Latency__ms_.png" alt="mistralai/Mistral-7B-v0.1 Mean Latency per token">
  </div>
 </center>
</div>

Please refer to the [quanto benchs](https://github.com/huggingface/quanto/tree/main/bench/) for detailed results for different model architectures and configurations.

## Integration in 🤗 transformers


Quanto library is seamlessly integrated in Hugging Face transformers library. You can quantize any model by passing a `QuantoConfig` to `from_pretrained`!

Currently you need to use the latest version of accelerate to make sure that the integration is fully compatible.

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

You can quantize the weights and / or activations in int8, float8, int4 or int2 by simply passing the correct argument in `QuantoConfig`. The activations can be either in int8 or float8 - note for float8 you need to have hardware that is compatible with float8 precision, otherwise quanto will silently upcast the weights and activations to torch.float32 or torch.float16 (depending on the original data type of the model) when we perform the matmul (only when the weight is quantized). Note if you try to use `float8` using MPS devices, `torch` will currently raise an error.

Quanto is device agnostic, meaning you can quantize your model regardless if you are on CPU / GPU / MPS (Apple Silicon) therefore you can run quantized models on any of these devices.

Quanto is also torch.compile friendly, you can quantize a model with quanto and call `torch.compile` to the model to compile it for faster generation. Note this feature might not work out of the box if dymanic quantization is involved (i.e. Quantization Aware Training or quantized activations enabled). Make sure to keep `activations=None` when creating your `QuantoConfig` in case you use the transformers integration.

Note it is also possible to quantize any model, regardless of the modality using quanto ! We demonstrate how to quantize `openai/whisper-large-v3` model in int8 using quanto.

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

Check out this [notebook](https://colab.research.google.com/drive/16CXfVmtdQvciSh9BopZUDYcmXCDpvgrT?usp=sharing#scrollTo=IHbdLXAg53JL) for a complete tutorial on how to properly use quanto with transformers integration !

## Implementation details

### Quantized tensors

At the heart of quanto are Tensor subclasses that corresponds to:
- the projection using a `scale` of a source Tensor into the optimal range for a given quantization type,
- the mapping of projected values to the destination type.

For floating-point destination types, the mapping is done by the native pytorch cast (i.e. `Tensor.to()`).

For integer destination types, the mapping is a simple rounding operation (i.e. `torch.round()`).

The goal of the projection is to increase the accuracy of the conversion by minimizing the number of:
- saturated values (i.e. mapped to the destination type min/max),
- zeroed values (because they are below the smallest number that can be represented by the destination type)

For efficiency, the projection is symmetric for `8-bit` quantization types, i.e. it is centered around zero.
Symmetric quantized Tensors are usually compatible with many standard operations.

For lower bitwidth quantization types, such as `int2` or `int4`, the projection is affine, i.e. it uses a `zeropoint` to shift the
projected values, which allows a better coverage of the quantization range. Affine quantized Tensors are typically harder to work with
and require custom operations.

### Quantized modules

Quanto provides a generic mechanism to replace torch modules (`torch.nn.Module`) by `quanto` modules that are able to process `quanto` tensors.

Quanto modules dynamically convert their `weight` parameter until a model is frozen, which slows down inference a bit but is
required if the model needs to be tuned (a.k.a Quantization Aware Training).

Module `bias` parameters are not quantized because they are much smaller than `weights` and quantized addition is hard to accelerate.

Activations are dynamically quantized using static scales (defaults to the range `[-1, 1]`). The model needs to be calibrated to evaluate
the best activation scales (using a momentum).

The following modules can be quantized:

- [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) (QLinear).
Weights are always quantized, and biases are not quantized. Inputs and outputs can be quantized.
- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) (QConv2D).
Weights are always quantized, and biases are not quantized. Inputs and outputs can be quantized.
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
Weights and biases are __not__ quantized. Outputs can be quantized.

### Custom operations

Thanks to the awesome pytorch dispach mechanism, [🤗 quanto](https://github.com/huggingface/quanto) provides implementations for
the most common functions used in 🤗 transformers or 🤗 diffusers models, allowing to use quantized Tensors without modifying
much the modeling code.

Most of these "dispatched" functions can be performed using combinations of standard pytorch operations.

Complex functions however require the definition of custom operations under the `torch.ops.quanto` namespace.

Examples of such operations are `dqmm` for W8A16 matrix multiplications and `udqmm` for lower bitwidth matrix multiplications.

### Post-training quantization optimizers

That feature is not available yet in [🤗 quanto](https://github.com/huggingface/quanto), but the library is versatile enough
to be compatible with most PTQ optimization algorithms.

The plan is to integrate the most popular algorithms in the most seamless possible way.

## Contributing to 🤗 quanto

Contributions to [🤗 quanto](https://github.com/huggingface/quanto) are very much welcomed, especially in the following areas:

- optimized kernels for [🤗 quanto](https://github.com/huggingface/quanto) operations targeting specific devices,
- PTQ optimizers,
- new dispatched operations for quantized Tensors.
