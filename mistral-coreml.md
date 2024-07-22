---
title: "WWDC 24: Running Mistral 7B with Core ML" 
thumbnail: /blog/assets/mistral-coreml/thumbnail.png
authors:
- user: pcuenq
- user: osanseviero
- user: reach-vb
- user: FL33TW00D-HF
---

# WWDC 24: Running Mistral 7B with Core ML

WWDC’ 24 is the moment Apple officially unveiled Apple Intelligence and
reiterated their commitment to efficient, private, and on-device AI.
During the keynote and the sessions that followed, they demonstrated
Apple Intelligence, which powers a huge array of AI-enhanced features
that show practical uses for everyday tasks. These are not
\*AI-for-the-sake-of-AI\* shiny demos. These are time-saving,
appropriate (and fun!) helpers that are deeply integrated with apps and
the OS, that also offer developers a number of ways to include these
features within their own apps.

Apple Intelligence features can only work this well
because of the vertically integrated software stack that harnesses
Apple Silicon's capabilities to the fullest. Apple also offers a platform for developers to run models on-device, known as
Core ML. This software stack allows you to run ML models across all 3
compute units (CPU, GPU & Neural Engine) available on Apple Silicon hardware.

In this blog post, we’ll be exploring some of the best new Core ML
features to replicate the Mistral 7B example Apple showcased in the
WWDC’24 [Deploy machine learning and AI models on-device with Core
ML](https://developer.apple.com/videos/play/wwdc2024/10161/)
session, where they use a fork of
[swift-transformers](https://github.com/huggingface/swift-transformers)
to run a state-of-the-art LLM on a Mac. This is a high-quality model
with more than 7 billion parameters that pushes the capabilities of
consumer hardware today. You can also check out WWDC’24 [Bring your
machine learning and AI models to Apple
silicon](https://developer.apple.com/videos/play/wwdc2024/10159/)
session, where part of the Mistral 7B conversion process is shown.

Let’s see what steps to take to run it as efficiently as possible, and
learn the new tools available in iOS 18 & macOS Sequoia.

This is what we’ll be building today:

<video controls title="Mistral 7B running with Core ML">
<source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mistral-coreml/swift-chat.mp4" type="video/mp4">
Video: Mistral 7B running with Core ML.
</video>

## TL;DR

By the end of this blog post, you will have learnt all the new goodies
accompanying the latest macOS release AND you will have successfully run
a 7B parameter model using less than 4GB of memory on your Mac.

Step 1: Clone the `preview` branch of the `swift-transformers` repo: git clone -b preview [`https://github.com/huggingface/swift-transformers`](https://github.com/huggingface/swift-transformers)
Step 2: Download the converted Core ML models from [`this Hugging Face repo`](https://huggingface.co/apple/mistral-coreml)
Step 3: Run inference using Swift: `swift run transformers "Best recommendations for a place to visit in Paris in August 2024:" --max-length 200 Mistral7B-CoreML/StatefulMistralInstructInt4.mlpackage`

## Best new Core ML features from WWDC’ 24

Here are some of the most impactful Core ML features from WWDC’ 24 we
will use to run Mistral 7B on a Mac.

### Swift Tensor

The first feature we want to highlight is an entirely new Swift type to
work with ML tensors. These are multi-dimensional data structures every
ML framework uses. Python developers working on ML are familiar with
`numpy` arrays or `torch` tensors, which provide convenient,
high-level interfaces to manipulate these large multi-dimensional
matrices easily. The new `MLTensor` type provides a high-level
abstraction that mimics the ones available in Python frameworks, greatly
simplifying working with tensor data in Swift.

Core ML already had multi-dimensional data types in the form of
[MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray)
and
[MLShapedArray](https://developer.apple.com/documentation/coreml/mlshapedarray).
However, they were only meant for data storage and simple operations
like wrapping your data and sending it as input to a Core ML model, or
unwrapping results from a Core ML model. However, *manipulating* tensor
data with these APIs is difficult. Only a few primitive operations are
provided, and you may have to write your own by accessing the underlying
storage as an opaque pointer to number data. This is time-consuming and
error-prone.

The new `Tensor` type provides a high-level abstraction that mimics
the ones available in Python frameworks, greatly simplifying working
with tensor data in Swift. Consider a language model like the one we
want to port to Core ML. Language models take in an input sequence of
tokens, and they output an estimation of the probabilities of all the
tokens in the vocabulary, meaning that tokens with a high probability
have a high chance of being plausible continuations of the input. The
application’s job is to select the best next token to append to the
sequence based on those probabilities. `Tensor` type makes it easy to
handle these operations without custom code.

[When we released swift-transformers](https://huggingface.co/blog/swift-coreml-llm),
we wrote a lot of code (later extended by the community, thanks! ❤️) to
help with input preparations (convert words to tokens) and output
post-processing. For example, check out [our softmax operation](https://github.com/huggingface/swift-transformers/blob/main/Sources/TensorUtils/Math.swift#L103)
using Accelerate. All this can be removed when using `MLTensor`, as
`softmax` is provided out of the box!

### Stateful Buffers

Before WWDC’ 24, a Core ML model was essentially a pure stateless
function where you provide inputs and return some outputs. However,
sometimes you need to keep a state that depends on previous
computations. The functional programming method for maintaining state is
to add an additional input/output pair. So, based on your inputs and
state, the model computes the output and the new state. There is nothing
wrong with this approach, and in fact, that’s the way high-performance
frameworks like JAX work.

However, there are practical limitations: the stateful data needs to be
sent to the model as an input and retrieved as an output every time you
call the model. If the stateful data is large, then all this going back
and forth increases overhead and slows things down. This is particularly
important for LLMs because you have to run many iterations to generate a
sequence. The performance bottleneck is usually your computer’s memory
bandwidth (i.e., how fast you can move things to your GPU and back).
Stateful models solve this problem by reserving a block of memory for
state data and keeping it on the GPU so you don’t have to send and
receive it every time you use the model.

Stateful buffers were introduced [in this WWDC’ 24 session](https://developer.apple.com/videos/play/wwdc2024/10161/?time=510)
using a toy example that is easy to understand but not representative of
practical uses with big models such as LLMs. An LLM performance trick
for transformers-based models is key-value caching (known as
kv-caching). As shown in the following illustration, it avoids costly
matrix multiplications in the crucial attention block by caching the
result of previous operations performed in previous steps. We won’t go
into details, but the takeaways are: kv-cache dramatically increases
performance, and it requires a large block of memory that is the perfect
candidate for using stateful buffers. Here is a [coremltools user guide](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
update about stateful models.

![stateful-buffer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mistral-coreml/stateful-buffer.png)

### New Quantization Techniques

In WWDC 23, we explored a very cool technique called palletization, and
we showed how it could help bring text-to-image models, [such as Stable
Diffusion](https://huggingface.co/blog/fast-diffusers-coreml), to Macs and iPhones.

Whilst these techniques allow you to reduce the size considerably, if
pushed too far, the impact on quality is drastic. Bigger models suffer
more from this, as the weight data has an extensive dynamic range.
Trying to create a small lookup table (LUT) that captures all possible
values becomes increasingly difficult. The solution introduced in WWDC
24 is to focus on a smaller portion of the data at a time, and create
multiple lookup tables for different areas of the same tensor.

![quantization-algorithm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/mistral-coreml/quantization-algorithm.png)

These methods (block-wise quantization) allow us to compress models to
as low as 4-bit precision. Instead of using 4 bytes (the size of a
`float32` number) to represent each model parameter, we can get away
with half a byte (a nibble) for each. This is an 8-fold reduction in
model size (minus some overhead to account for the block-wise
quantization tables), or 4 times smaller when compared to `float16`
precision.

### Multifunction Support

We won’t use this feature for this example but we wanted to mention it
here as it was introduced at WWDC 24, and we will be showcasing it in
some upcoming work. Multifunction support essentially allows you to
package LoRA adapters into generative models to use the same model (with
a small set of additional parameters, called adapters) for different
tasks. LoRA is the preferred community technique for large model
fine-tuning. In diffusion models, for example, you can use LoRA to
generate images with different styles, such as photorealistic or
cartoonish. We believe LoRA is part of the solution that powers Apple’s
Genmoji implementation. For language models, LoRA adapters can be used
to adapt a generic LLM to specific tasks or domains.

To read more about LoRA, you can check [this post.](https://huggingface.co/blog/lora)

To read more about Multifunction, you can check out Apple coremltools
user guide [here](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html).

## Converting Mistral 7B to Core ML

The single most important component for running a large language model
efficiently is the kv-cache. As mentioned above, this is a great
candidate for [the new stateful model feature](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
released at WWDC’ 24. Models in the transformers library already use
efficient attention implementations that rely heavily on kv-caching.
However, the default implementations are optimized for Nvidia GPUs, and
this hardware has a different set of constraints than Apple Silicon
does. In the case of Core ML, we need to pre-allocate the full cache
buffer beforehand and ensure that each time we call the model, we update
the buffer in place. This avoids inefficient memory allocations and
tensor concatenations and is also a requirement for Core ML stateful
buffers.

To achieve this goal, we have to use a different attention
implementation that considers these factors. This requires modifying the
transformers modeling code for the Mistral architecture, and it’s done
in [this fragment of code](https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/export.py#L121).

Note: If you want to follow along and replicate the conversion (or
convert another Mistral-based model, like a different fine-tune), you
can use [this script](https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/export.py)
to run all the conversion steps.

### Tracing & Conversion

The first step is to load the model. We’ll use the patched
implementation with the in-place cache method.

```python
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
torch_model = StatefulMistralForCausalLM(MODEL_ID)
torch_model.eval()
```

Before running Core ML conversion, we need to trace the model with
example inputs. This process records the tensor operations performed on
those inputs, and the traced graph will be translated to Core ML
operations during conversion. We use sample inputs to trace the model;
we don’t need real data.

```python
input_ids = torch.zeros((1, 2), dtype=torch.int32)
causal_mask = torch.zeros((1, 1, 2, 5), dtype=torch.float32)

traced_model = torch.jit.trace(torch_model, [input_ids, causal_mask])

```

The input to a language model is a sequence of tokens of varying length.
We’ll allow the input to grow from a single token to a maximum context
length of 2048. We can use
[coremltools](https://github.com/apple/coremltools) range
dimensions to specify these bounds.

```python
query_length = ct.RangeDim(lower_bound=1, upper_bound=2048, default=1)
end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=2048, default=1)

inputs = [
    ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
    ct.TensorType(shape=(1, 1, query_length, end_step_dim), dtype=np.float16, name="causalMask"),
]

outputs = [ct.TensorType(dtype=np.float16, name="logits")]

```

In addition to the sequence tokens (called `inputIds` in the example
above), there’s another input called `causalMask`, which specifies the
tokens the model needs to pay attention to. This is mostly used when
generating multiple sequences at the same time using batching. Check out
how these inputs are used in an [example runner
here](https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/generate.py#L29-L42).

In this situation, all the input sequences inside a batch must have the
same length, so we use padding tokens and the causal mask to tell the
model that the padding tokens are not to be considered as inputs.

### State Preparation

The PyTorch modeling code uses `keyCache` and `valueCache` as the
names of the cache buffers to hold the kv-cache. Those blocks are
allocated for the maximum context length (2048). We use `coremltools`'
new
[StateType](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html#statetype)
to specify that those blocks must be converted to a stateful Core ML
buffer during conversion.

```python
# Specify kv-cache states by using `StateType`.

states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
        name="keyCache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
        name="valueCache",
    ),
]

```

### Core ML Conversion

To convert the model to Core ML, we need to specify the input and output
types, as well as the states. The converted model will use `float16`
precision because that’s what we specified for the input data. We also
need to indicate the minimum deployment target as iOS18, as that’s where
these features are available. (We can also use `macOS15`, which refers
to the same conversion target.)

```python
mlmodel_fp16 = ct.convert(
    traced_model,
    inputs=inputs,
    states=states,
    outputs=outputs,
    minimum_deployment_target=ct.target.iOS18,
    skip_model_load=True,
)
```

### Model Compression

Using the new block-size quantization strategies described above, we use
4-bit linear quantization with block size 32. This will greatly reduce
model size and make the model run faster. Even though computation will
still be performed in `float16`, weights are transferred in 4-bit mode
and decompressed on the fly, which is more efficient than transferring a
large amount of 16-bit weights.

The quantization parameters are configured as follows:

```python
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",
    granularity="per_block",
    block_size=32,
)
config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
```

Let’s use that configuration to quantize the model. The following line
will take a few minutes to run:

```python
mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)

mlmodel_int4.save("StatefulMistral7BInstructInt4.mlpackage")

```

There’s a final step after conversion and quantization are done. We need
to include a piece of additional metadata that indicates the model
identifier we used (`mistralai/Mistral-7B-Instruct-v0.3`). The Swift
code will use this to download the tokenizer files from the Hub.
Tokenization is converting text data to the numerical representations
used by models, and it’s different for every model.

```python
mlmodel_int4._spec.description.metadata.userDefined.update({
    "co.huggingface.exporters.name": MODEL_ID
})

```

The generated model is a `mlpackage` of about 3.8G, compared with the
14G that a `float16` conversion would produce. [You can find it
here on the
Hub.](https://huggingface.co/apple/mistral-coreml/tree/main)

## Running Mistral 7B with Swift

If you followed the steps above or downloaded the model from the Hub,
you can run it locally using the `preview` branch of
`swift-transformers`. Apple engineers contributed it to the project,
including the following important features:

-   Full `Tensor` support, which greatly simplifies pre- and
    post-processing tasks, and allows us to delete many lines of
    low-level, confusing and fragile code.

-   Support for the Swift counterpart of the Stateful API.

Since adopting these features is a breaking change and requires iOS 18
or macOS 15, we’ll keep them in a `preview` branch for now.

To run the model from the command line, please first clone the `preview`
branch from the GitHub repo:

```bash 
    git clone -b preview https://github.com/huggingface/swift-transformers
```

And then run the CLI to test the model:

```bash
#to run in release mode, pass -c release
swift run transformers "Best recommendations for a place to visit in Paris in August 2024:" --max-length 128 Examples/Mistral7B/StatefulMistral7BInstructInt4.mlpackage
```

For easier testing, you can also use `swift-chat`, a simple app we
wrote to show how to integrate the `swift-transformers` package
inside. You have to use the `preview` branch as well. An example of
`swift-chat` running the converted Mistral model was shown at the
beginning of this post.

## Running Mistral 7B with Python

For those of you who are more familiar with Python, it’s just as easy!

```bash
python3 generate.py Examples/Mistral7B/StatefulMistral7BInstructInt4.mlpackage --prompt "Best recommendations for a place to visit in Paris in August 2024:"
```

coremltools makes it just as easy to run Core ML models with Python.

## What's Next? 

We are extremely excited about the progress in [Core ML](https://developer.apple.com/documentation/coreml/) and
[coremltools](https://github.com/apple/coremltools) this year,
and we are looking forward to seeing lots of third-party apps leveraging
ML models to solve real tasks people need. On our side, we are committed
to making this as easy as possible so developers can concentrate on
creating cool apps. There are a few things on our drawing board:

-   The model updates presented here are excellent for GPUs on Mac
    computers. Core ML can use the Neural Engine, which is particularly
    efficient on iPhones. Getting the most performance out of the Neural
    Engine requires some additional adaptations, which we plan to carry
    out on a few example models. This work will be based on the
    learnings discussed in this [2022 (and still very relevant) article by Apple](https://machinelearning.apple.com/research/neural-engine-transformers).
    We won’t run Mistral 7B on iPhone, but there are several smaller
    models, like Apple’s OpenELM or DCLM that make for great
    candidates to explore!

-   The code presented here is highly experimental. As summer goes on,
    we plan to adopt these methods and incorporate them into
    `exporters`, a Python tool designed to convert transformers models
    to Core ML. Hopefully, you’ll soon be able to convert many
    interesting model architectures very easily.

-   We’ll keep working on the `preview` branch of
    `swift-transformers` to incorporate new features or API changes as
    they are released. If you are interested, keep an eye on it!

## How can you help?

The tools released by Apple in WWDC help us on our long-term goal to
make AI easy and accessible to all, and we’d love to see where you can
take them. The example we showed is experimental, but you can use it to
convert any Mistral fine-tune to Core ML – please let us know if you do!
If you want to try other model architectures, please feel free to open
issues or PRs to the `preview` branch of `swift-transformers` –
we’ll try to help you get going!

There’s never been a better time than today to apply your creativity to
solve problems that interest you! Go try things, have fun, and tell us
how we can help.


