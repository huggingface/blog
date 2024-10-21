---
title: “Llama 3.2 in Keras” 
thumbnail: /blog/keras_llama_32/thumbnail.gif
authors:
- user: martin-gorner
---


# Llama 3.2 in Keras

This is going to be the shortest blog post ever.

> **Question**: *Llama 3.2 landed two weeks ago on Hugging Face / Transformers. When will it be available in Keras?*

> **Answer**: *It has been working from day 1 😀. There is nothing to wait for.*

Yes, Keras Llama3 can be loaded from any standard (i.e. safetensor) Hugging Face checkpoint, including the 3.2 checkpoints. If a conversion is required, it happens on the fly. Try this:

```Python
!pip install keras_hub

from keras_hub import models.Llama3CausalLM
model = Llama3CausalLM.from_preset("hf://meta-llama/Llama-3.2-1B-Instruct", dtype="float16")
model.generate("Hi there!")
```

Here is a [Colab](https://colab.research.google.com/drive/1Zz4wTCCYV3BqtFLroNDLqmUhJZJiqpkK?usp=sharing) to try this out. Enjoy! 🤗

---

OK, OK, I'm being told that if I want to publish a blog post, I have to fill the space. Here are a couple of additional things to know about Keras.

## Keras is multi-backend

Keras is the time-tested modeling library for JAX, PyTorch and TensorFlow. You might have noticed this line in the [demo Colab](https://colab.research.google.com/drive/1Zz4wTCCYV3BqtFLroNDLqmUhJZJiqpkK?usp=sharing):

```Python
import os
os.environ["KERAS_BACKEND"] = "jax" # or "torch", or "tensorflow"
```

It has to appear before `import keras` and controls if the model is running on JAX, PyTorch or TensorFlow. Very handy to try your favorite models on JAX with XLA compilation 🚀.

## What is keras-hub?

Keras is a modeling library and [keras-hub](https://keras.io/keras_hub/) is its collection of pre-trained models. It was previously called [KerasNLP](https://keras.io/keras_nlp/) and [KerasCV](https://keras.io/keras_cv/). The [rename](https://github.com/keras-team/keras-hub/issues/1831) is in progress. It has all the popular pre-trained models (Llama3, Gemma, StableDiffusion, Segment Anything, ...) with their canonical implementation in Keras.

## LLMs in Keras come "batteries included"

I mean, "tokenizer included". `model.generate()` just works on strings:
```
model.generate("Hi there!")
```

Same thing for training:

```
model.fit(strings) # list or dataset of input strings
```

## Chatting with an LLM

Instruction-tuned variants of popular LLMs can be used for turn-by-turn conversations. Here, Llama-3.2-1B-Instruct understands the following conversation tagging (see [meta docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#-instruct-model-prompt-)).

```
<|start_header_id|>system<|end_header_id|>You  are a helpful assistant<|eot_id|>\n
\n
<|start_header_id|>user<|end_header_id|>Hello_<|eot_id|>\n
\n
<|start_header_id|>assistant<|end_header_id|>\n
\n
```
The conversation, once formatted in this way, can be fed directly to `model.generate()`.

For convenience, the [demo Colab](https://colab.research.google.com/drive/1Zz4wTCCYV3BqtFLroNDLqmUhJZJiqpkK?usp=sharing) implements a helper class called `ChatState` that does the necessary string concats automatically.

## Keras has a built-in trainer

TODO

## Lower level access: Tokenizer, Backbone

If you don't like "batteries included" and want to get to the underlying tokenizer and model, they are easily accessible:

```
# tokenizer
model.preprocessor.tokenizer

# the model itself
model.backbone

# You can even load them separately from the same preset
backbone = keras_hub.models.Llama3CausalLM.from_preset("hf://meta-llama/Llama-3.2-1B-Instruct", dtype="float16")
tokenizer = keras_hub.models.Llama3Tokenizer.from_preset("hf://meta-llama/Llama-3.2-1B-Instruct")
```

## Wait, Tokenizer, Preprocessor? I'm confused

The Tokenizer just transforms text into integer vectors. Here "Hello" translates into a single token:

```
tokenizer("Hello")
> Array([9906], dtype=int32)
```

The Preprocessor is a catch-all concept for doing all the data transformations a model requires. This could be, for example, image resizing or augmentation for tasks involving images, or text tokenization like here for a text model. For the CausalLM task, the preprocessor takes care of three additional details:
* adding the text start and text end tokens expected by the model
* padding the token sequences and generating a mask
* generating "expected outputs" for training and fine-tuning. For CausalLM tasks this is the input string shifted by one.

```
tokens = model.preprocessor("Hello")

tokens[0] # 128000 and 128009 are the start and end text tokens
> {'token_ids': Array([128000,   9906, 128009, 0, 0, 0], dtype=int32), 'padding_mask': Array([True, True, True, False, False, False], dtype=bool)}

tokens[1] # input sequence shifted by one
> [9906, 128009, 0, 0, 0, 0]

# feeding the model manually
model.backbone(model.preprocessor(["Hello", "Hi!"])[0]) # raw logits as output
> [[[ 0.9805   0.1664   0.625   ... -0.834   -0.264    0.05203]
  ...]]

# More typically you would use Keras built-in functions model.generate, model.fit, model.predict, model.evaluate
```

## You can upload to Hub too

TODO

## Distributed inference or training

TODO