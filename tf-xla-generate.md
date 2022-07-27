---
title: 'Faster Text Generation with TensorFlow and XLA'
thumbnail: /blog/assets/91_tf_xla_generate/thumbnail.png
---

<h1>
    Faster Text Generation with TensorFlow and XLA
</h1>

<div class="blog-metadata">
    <small>Published July 27, 2022.</small>
    <a target="_blank" class="btn no-underline text-sm mb-5 font-sans" href="https://github.com/huggingface/blog/blob/main/tf_xla_generate.md">
        Update on GitHub
    </a>
</div>

<div class="author-card">
    <a href="/joaogante">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1641203017724-noauth.png?w=200&h=200&f=face">
        <div class="bfc">
            <code>joaogante</code>
            <span class="fullname">João Gante</span>
        </div>
    </a>
</div>

<em>TL;DR</em>: Text Generation on 🤗 `transformers` using TensorFlow can now be compiled with XLA. It is up to 100x
faster than before, and [even faster than PyTorch](https://huggingface.co/spaces/joaogante/tf_xla_generate_benchmarks)
-- check the colab below!
<a target="_blank" href="https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Text Generation

As the quality of large language models increased, so did our expectations of what those models could do. Especially
since the release of OpenAI's [GPT-2](https://openai.com/blog/better-language-models/), models with text
generation capabilities have been in the spotlight. And for legitimate reasons -- these models can be used to
summarize, translate, and they even have demonstrated zero-shot learning capabilities on some language tasks.
This blog post will show how to take the most of this technology with TensorFlow.

The 🤗 `transformers` library started with NLP models, so it is natural that text generation is of utmost
importance to us.
It is part of Hugging Face democratization efforts to ensure it is accessible, easily controllable, and efficient.
There is a previous [blog post](https://huggingface.co/blog/how-to-generate) about the different types of text
generation. Nevertheless, below there's a quick recap of the core functionality -- feel free to
[skip it](#tensorflow-and-xla) if you're
familiar with our `generate` function and want to jump straight into TensorFlow's specificities.

Let's start with the basics. Text generation can be deterministic or stochastic, depending on the
`do_sample` flag. By default it's set to `False`, causing the output to be deterministic, which is also known as
Greedy Decoding.
When it's set to `True`, also known as Sampling, the output will be stochastic, but you can still
obtain reproducible results through the `seed` argument (with the same format as in [stateless TensorFlow random
number generation](https://www.tensorflow.org/api_docs/python/tf/random/stateless_categorical#args)).
As a rule of thumb, you want deterministic generation if you wish
to obtain factual information from the model and stochastic generation if you're aiming at more creative outputs.

```python
# Requires transformers >= 4.21.0;
# Sampling outputs may differ, depending on your hardware.
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
inputs = tokenizer(["TensorFlow is"], return_tensors="tf")

generated = model.generate(**inputs, do_sample=True, seed=(42, 0))
print("Sampling output: ", tokenizer.decode(generated[0]))
# > Sampling output: TensorFlow is a great learning platform for learning about
# data structure and structure in data science..
```

Depending on the target application, longer outputs might be desirable. You can control the length of the generation
output with `max_new_tokens`, keeping in mind that longer generations will require more resources.

```python
generated = model.generate(
    **inputs, do_sample=True, seed=(42, 0), max_new_tokens=5
)
print("Limiting to 5 new tokens:", tokenizer.decode(generated[0]))
# > Limiting to 5 new tokens: TensorFlow is a great learning platform for
generated = model.generate(
    **inputs, do_sample=True, seed=(42, 0), max_new_tokens=30
)
print("Limiting to 30 new tokens:", tokenizer.decode(generated[0]))
# > Limiting to 30 new tokens: TensorFlow is a great learning platform for
# learning about data structure and structure in data science................
```

Sampling has a few knobs you can play with to control randomness. The most important is `temperature`, which sets the overall entropy
of your output -- values below `1.0` will prioritize sampling tokens with a higher likelihood, whereas values above `1.0`
do the opposite. Setting it to `0.0` reduces the behavior to Greedy Decoding, whereas very large values approximate
uniform sampling.

```python
generated = model.generate(
    **inputs, do_sample=True, seed=(42, 0), temperature=0.7
)
print("Temperature 0.7: ", tokenizer.decode(generated[0]))
# > Temperature 0.7: TensorFlow is a great way to do things like this........
generated = model.generate(
    **inputs, do_sample=True, seed=(42, 0), temperature=1.5
)
print("Temperature 1.5: ", tokenizer.decode(generated[0]))
# > Temperature 1.5: TensorFlow is being developed for both Cython and Bamboo.
# On Bamboo...
```

Contrarily to Sampling, Greedy Decoding will always pick the most likely token at each iteration of generation.
However, it often results in sub-optimal outputs. You can increase the quality of the results through the `num_beams`
argument. When it is larger than `1`, it triggers Beam Search, which continuously explores high-probability sequences.
This exploration comes at the cost of additional resources and computational time.

```python
generated = model.generate(**inputs, num_beams=2)
print("Beam Search output:", tokenizer.decode(generated[0]))
# > Beam Search output: TensorFlow is an open-source, open-source,
# distributed-source application framework for the
```

Finally, when running Sampling or Beam Search, you can use `num_return_sequences` to return several sequences. For
Sampling it is equivalent to running generate multiple times from the same input prompt, while for Beam Search it
returns the highest scoring generated beams in descending order.

```python
generated = model.generate(**inputs, num_beams=2, num_return_sequences=2)
print(
    "All generated hypotheses:",
    "\n".join(tokenizer.decode(out) for out in generated)
)
# > All generated hypotheses: TensorFlow is an open-source, open-source,
# distributed-source application framework for the
# > TensorFlow is an open-source, open-source, distributed-source application
# framework that allows
```

The basics of text generation, as you can see, are straightforward to control. However, there are many options
not covered in the examples above, and it's encouraged to read the
[documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_tf_utils.TFGenerationMixin.generate)
for advanced use cases.
Sadly, when you run `generate` with TensorFlow, you might notice that it takes a while to execute.
If your target application expects low latency or a large amount of input prompts, running text generation with
TensorFlow looks like an expensive endeavour. 😬

Fear not, for the remainder of this blog post aims to demonstrate that one line of code can make a drastic improvement.
If you'd rather jump straight into action,
[the colab](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb)
has an interactive example you can fiddle with!

## TensorFlow and XLA

[XLA](https://www.tensorflow.org/xla), or Accelerated Linear Algebra, is a compiler originally developed to accelerate
TensorFlow models. Nowadays, it is also the compiler behind [JAX](https://github.com/google/jax), and it can even
be [used with PyTorch](https://huggingface.co/blog/pytorch-xla). Although the word "compiler" might sound daunting for
some, XLA is simple to use with TensorFlow -- it comes packaged inside the `tensorflow` library, and it can be
triggered with the `jit_compile` argument in any graph-creating function.

For those of you familiar with TensorFlow 1 🧓, the concept of a TensorFlow graph comes naturally, as it was the only
mode of operation. First, you defined the operations in a declarative fashion to create a graph. Afterwards, you could
pipe inputs through the graph and observe the outputs. Fast, efficient, but painful to debug. With TensorFlow 2 came
Eager Execution and the ability to code the models imperatively -- the TensorFlow team explains the difference in more
detail in [their blog post](https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html).

Hugging Face writes their TensorFlow models with Eager Execution in mind. Transparency is a core value, and being able
to inspect the model internals at any point is very benefitial to that end. However, that does mean that some uses of
the models do not benefit from the graph mode performance advantages out of the box (e.g. when calling `model(args)`).

Fortunately, the TensorFlow team has users like us covered 🥳! Wrapping a function containing TensorFlow code with
[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) will attempt to convert it into a graph when
you call the wrapped function. If you're training a model, calling `model.compile()` (without `run_eagerly=True`) does
precisely that wrapping, so that you benefit from graph mode when you call `model.fit()`. Since `tf.function` can be
used in any function containing TensorFlow code, it means you can use it on functions that go beyond model inference,
creating a single optimized graph.

Now that you know how to create TensorFlow graphs, compiling them with XLA is straightforward -- simply add `jit_compile=True`
as an argument to the functions mentioned above (`tf.function` and `tf.keras.Model.compile`). Assuming everything went well
(more on that below) and that you are using a GPU or a TPU, you will notice that the first call will take a while, but
that the remaining ones are much, much faster. Here's a simple example of a function that performs model inference and some post-processing of its outputs:

```python
# Note: execution times are deeply dependent on hardware -- a 3090 was used here.
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
inputs = tokenizer(["TensorFlow is"], return_tensors="tf")

def most_likely_next_token(inputs):
    model_output = model(inputs)
    return tf.argmax(model_output.logits[:, -1, :], axis=-1)

print("Calling regular function with TensorFlow code...")
most_likely_next_token(inputs)
# > Execution time -- 48.8 ms
```

In one line, you can create an XLA-accelerated function from the function above.

```python
xla_most_likely_next_token = tf.function(most_likely_next_token, jit_compile=True)

print("Calling XLA function... (for the first time -- will be slow)")
xla_most_likely_next_token(inputs)
# > Execution time -- 3951.0 ms
print("Calling XLA function... (for the second time -- will be fast)")
xla_most_likely_next_token(inputs)
# > Execution time -- 1.6 ms
```

## Text Generation using TensorFlow with XLA

As with any optimization procedure, there is no free lunch -- XLA is no exception. From the perspective of a text
generation user, there is only one technical aspect that you need to keep in mind. Without digging too much into
[details](https://www.tensorflow.org/guide/function#rules_of_tracing), XLA used in this fashion does just-in-time (JIT)
compilation of a `tf.function` when you call it, which relies on polymorphism.

When you compile a function this way, XLA keeps track of the shape and type of every tensor, as well as the data of
every non-tensor function input. The function is compiled to a binary, and every time it is called with the same tensor
shape and type (with ANY tensor data) and the same non-tensor arguments, the compiled function can be reused.
Contrarily, if you call the function with a different shape or type in an input tensor, or if you use a different
non-tensor argument, then a new costly compilation step will take place. Summarized in a simple example:

```python
# Note: execution times are deeply dependent on hardware -- a 3090 was used here.
import tensorflow as tf

@tf.function(jit_compile=True)
def max_plus_constant(tensor, scalar):
    return tf.math.reduce_max(tensor) + scalar

# Slow: XLA compilation will kick in, as it is the first call
max_plus_constant(tf.constant([0, 0, 0]), 1)
# > Execution time -- 520.4 ms

# Fast: Not the first call with this tensor shape, tensor type, and exact same
# non-tensor argument
max_plus_constant(tf.constant([1000, 0, -10]), 1)
# > Execution time -- 0.6 ms

# Slow: Different tensor type
max_plus_constant(tf.constant([0, 0, 0], dtype=tf.int64), 1)
# > Execution time -- 27.1 ms

# Slow: Different tensor shape
max_plus_constant(tf.constant([0, 0, 0, 0]), 1)
# > Execution time -- 25.5 ms

# Slow: Different non-tensor argument
max_plus_constant(tf.constant([0, 0, 0]), 2)
# > Execution time -- 24.9 ms
```

In practice, for text generation, it simply means the input should be padded to a multiple of a certain length (so it
has a limited number of possible shapes), and that using different options will be slow for the first time you use
them. Let's see what happens when you naively call generation with XLA.

```python
# Note: execution times are deeply dependent on hardware -- a 3090 was used here.
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# Notice the new argument, `padding_side="left"` -- decoder-only models, which can
# be instantiated with TFAutoModelForCausalLM, should be left-padded, as they
# continue generating from the input prompt.
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", padding_side="left", pad_token="</s>"
)
model = TFAutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
input_1 = ["TensorFlow is"]
input_2 = ["TensorFlow is a"]

# One line to create a XLA generation function
xla_generate = tf.function(model.generate, jit_compile=True)

# Calls XLA generation without padding
tokenized_input_1 = tokenizer(input_1, return_tensors="tf")  # length = 4
tokenized_input_2 = tokenizer(input_2, return_tensors="tf")  # length = 5
print(f"`tokenized_input_1` shape = {tokenized_input_1.input_ids.shape}")
print(f"`tokenized_input_2` shape = {tokenized_input_2.input_ids.shape}")

print("Calling XLA generation with tokenized_input_1...")
print("(will be slow as it is the first call)")
start = time.time_ns()
xla_generate(**tokenized_input_1)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 9565.1 ms

print("Calling XLA generation with tokenized_input_2...")
print("(has a different length = will trigger tracing again)")
start = time.time_ns()
xla_generate(**tokenized_input_2)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 6815.0 ms
```

Oh no, that's terribly slow! A solution to keep the different combinations of shapes in check is through padding,
as mentioned above. The tokenizer classes have a `pad_to_multiple_of` argument that can be used to achieve a balance
between accepting any input length and limiting tracing.

```python
padding_kwargs = {"pad_to_multiple_of": 8, "padding": True}
tokenized_input_1_with_padding = tokenizer(
    input_1, return_tensors="tf", **padding_kwargs
)  # length = 8
tokenized_input_2_with_padding = tokenizer(
    input_2, return_tensors="tf", **padding_kwargs
)  # length = 8
print(
    "`tokenized_input_1_with_padding` shape = ",
    f"{tokenized_input_1_with_padding.input_ids.shape}"
)
print(
    "`tokenized_input_2_with_padding` shape = ",
    f"{tokenized_input_2_with_padding.input_ids.shape}"
)

print("Calling XLA generation with tokenized_input_1_with_padding...")
print("(slow, first time running with this length)")
start = time.time_ns()
xla_generate(**tokenized_input_1_with_padding)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 6815.4 ms

print("Calling XLA generation with tokenized_input_2_with_padding...")
print("(will be fast!)")
start = time.time_ns()
xla_generate(**tokenized_input_2_with_padding)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 19.3 ms
```

That's much better, successive generation calls performed this way will be orders of magnitude faster than before.
Keep in mind that trying new generation options, at any point, will trigger tracing.

```python
print("Calling XLA generation with the same input, but with new options...")
print("(slow again)")
start = time.time_ns()
xla_generate(**tokenized_input_1_with_padding, num_beams=2)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
# > Execution time -- 9644.2 ms
```

From a developer perspective, relying on XLA implies being aware of a few additional nuances. XLA shines when the size
of the data structures are known in advance, such as in model training. On the other hand, when their dimensions are
impossible to determine or certain dynamic slices are used, XLA fails to compile. Modern implementations of text
generation are auto-regressive, whose natural behavior is to expand tensors and to abruptly interrupt some operations
as it goes -- in other words, not XLA-friendly by default.
We have [rewritten our entire TensorFlow text generation codebase](https://github.com/huggingface/transformers/pull/17857)
to vectorize operations and use fixed-sized
structures with padding. Our NLP models were also modified to correctly use their positional embeddings in the
presence of padded structures. The result should be invisible to TensorFlow text generation users, except for the
availability of XLA compilation.

## Benchmarks and Conclusions

Above you saw that you can convert TensorFlow functions into a graph and accelerate them with XLA compilation.
Current forms of text generation are simply an auto-regressive functions that alternate between a model forward pass
and some post-processing, producing one token per iteration. Through XLA compilation, the entire process gets
optimized, resulting in faster execution. But how much faster? The [Gradio demo below](https://huggingface.co/spaces/joaogante/tf_xla_generate_benchmarks) contains some benchmarks
comparing Hugging Face's text generation on multiple GPU models for the two main ML frameworks, TensorFlow and PyTorch.

<div class="hidden xl:block">
<div style="display: flex; flex-direction: column; align-items: center;">
<iframe src="https://hf.space/embed/joaogante/tf_xla_generate_benchmarks/+" frameBorder="0" width="1200px" height="760px" title="Gradio app" allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking" sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"></iframe>
</div>
</div>

If you explore the results, two conclusions become quickly visible:

1. As this blog post has been building up to here, TensorFlow text generation is much faster when XLA is used. We are
talking about speedups larger than 100x in some cases, which truly demonstrates the power of a compiled graph 🚀
2. TensorFlow text generation with XLA is the fastest option in the vast majority of cases, in some of them by as
much as 9x faster, debunking the myth that PyTorch is the go-to framework for serious NLP tasks 💪

Give [the colab](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb)
a go, and enjoy the power of text generation supercharged with XLA!
