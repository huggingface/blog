---
title: "Releasing Swift Transformers: Run On-Device LLMs in Apple Devices"
thumbnail: /blog/assets/swift-coreml-llm/thumbnail.png
authors:
- user: pcuenq
---

# Releasing Swift Transformers: Run On-Device LLMs in Apple Devices


I have a lot of respect for iOS/Mac developers. I started writing apps for iPhones in 2007, when not even APIs or documentation existed. The new devices adopted some unfamiliar decisions in the constraint space, with a combination of power, screen real estate, UI idioms, network access, persistence, and latency that was different to what we were used to before. Yet, this community soon managed to create top-notch applications that felt at home with the new paradigm.

I believe that ML is a new way to build software, and I know that many Swift developers want to incorporate AI features in their apps. The ML ecosystem has matured a lot, with thousands of models that solve a wide variety of problems. Moreover, LLMs have recently emerged as almost general-purpose tools – they can be adapted to new domains as long as we can model our task to work on text or text-like data. We are witnessing a defining moment in computing history, where LLMs are going out of research labs and becoming computing tools for everybody.

However, using an LLM model such as Llama in an app involves several tasks which many people face and solve alone. We have been exploring this space and would love to continue working on it with the community. We aim to create a set of tools and building blocks that help developers build faster.

Today, we are publishing this guide to go through the steps required to run a model such as Llama 2 on your Mac using Core ML. We are also releasing alpha libraries and tools to support developers in the journey. We are calling all Swift developers interested in ML – is that _all_ Swift developers? – to contribute with PRs, bug reports, or opinions to improve this together.

Let's go!

<p align="center">
  <video controls title="Llama 2 (7B) chat model running on an M1 MacBook Pro with Core ML">
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/swift-transformers/llama-2-7b-chat.mp4" type="video/mp4">
  <em>Video: Llama 2 (7B) chat model running on an M1 MacBook Pro with Core ML.</em>
</p>

## Released Today

- [`swift-transformers`](https://github.com/huggingface/swift-transformers), an in-development Swift package to implement a transformers-like API in Swift focused on text generation. It is an evolution of [`swift-coreml-transformers`](https://github.com/huggingface/swift-coreml-transformers) with broader goals: Hub integration, arbitrary tokenizer support, and pluggable models.
- [`swift-chat`](https://github.com/huggingface/swift-chat), a simple app demonstrating how to use the package.
- An updated version of [`exporters`](https://github.com/huggingface/exporters), a Core ML conversion package for transformers models.
- An updated version of [`transformers-to-coreml`](https://huggingface.co/spaces/coreml-projects/transformers-to-coreml), a no-code Core ML conversion tool built on `exporters`.
- Some converted models, such as [Llama 2 7B](https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml) or [Falcon 7B](https://huggingface.co/tiiuae/falcon-7b-instruct/tree/main/coreml), ready for use with these text generation tools.

## Tasks Overview

When I published tweets showing [Falcon](https://twitter.com/pcuenq/status/1664605575882366980) or [Llama 2](https://twitter.com/pcuenq/status/1681404748904431616) running on my Mac, I got many questions from other developers asking how to convert those models to Core ML, because they want to use them in their apps as well. Conversion is a crucial step, but it's just the first piece of the puzzle. The real reason I write those apps is to face the same problems that any other developer would and identify areas where we can help. We'll go through some of these tasks in the rest of this post, explaining where (and where not) we have tools to help.

- [Conversion to Core ML](#conversion-to-core-ml). We'll use Llama 2 as a real-life example.
- [Optimization](#optimization) techniques to make your model (and app) run fast and consume as little memory as possible. This is an area that permeates across the project and there's no silver-bullet solution you can apply.
- [`swift-transformers`](#swift-transformers), our new library to help with some common tasks.
    - [Tokenizers](#tokenizers). Tokenization is the way to convert text input to the actual set of numbers that are processed by the model (and back to text from the generated predictions). This is a lot more involved than it sounds, as there are many different options and strategies.
    - [Model and Hub wrappers](#model-and-hub-wrappers). If we want to support the wide variety of models on the Hub, we can't afford to hardcode model settings. We created a simple `LanguageModel` abstraction and various utilities to download model and tokenizer configuration files from the Hub.
    - [Generation Algorithms](#generation-algorithms). Language models are trained to predict a probability distribution for the next token that may appear after a sequence of text. We need to call the model multiple times to generate text output and select a token at each step. There are many ways to decide which token we should choose next.
    - [Supported Models](#supported-models). Not all model families are supported (yet).
- [`swift-chat`](#swift-chat). This is a small app that simply shows how to use `swift-transformers` in a project.
- [Missing Parts / Coming Next](#missing-parts--coming-next). Some stuff that's important but not yet available, as directions for future work.
- [Resources](#resources). Links to all the projects and tools.


## Conversion to Core ML

Core ML is Apple's native framework for Machine Learning, and also the name of the file format it uses. After you convert a model from (for example) PyTorch to Core ML, you can use it in your Swift apps. The Core ML framework automatically selects the best hardware to run your model on: the CPU, the GPU, or a specialized tensor unit called the Neural Engine. A combination of several of these compute units is also possible, depending on the characteristics of your system and the model details.

To see what it looks like to convert a model in real life, we'll look at converting the recently-released Llama 2 model. The process can sometimes be convoluted, but we offer some tools to help. These tools won't always work, as new models are being introduced all the time, and we need to make adjustments and modifications.

Our recommended approach is:

1. Use the [`transformers-to-coreml`](https://huggingface.co/spaces/coreml-projects/transformers-to-coreml) conversion Space:

This is an automated tool built on top of `exporters` (see below) that either works for your model, or doesn't. It requires no coding: enter the Hub model identifier, select the task you plan to use the model for, and click apply. If the conversion succeeds, you can push the converted Core ML weights to the Hub, and you are done!

You can [visit the Space](https://huggingface.co/spaces/coreml-projects/transformers-to-coreml) or use it directly here:

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.23.0/gradio.js"></script>
<gradio-app theme_mode="light" space="coreml-projects/transformers-to-coreml"></gradio-app>


2. Use [`exporters`](https://github.com/huggingface/exporters), a Python conversion package built on top of Apple's `coremltools` (see below).

This library gives you a lot more options to configure the conversion task. In addition, it lets you create your own [conversion configuration class](https://github.com/huggingface/exporters#overriding-default-choices-in-the-configuration-object), which you may use for additional control or to work around conversion issues.

3. Use [`coremltools`](https://github.com/apple/coremltools), Apple's conversion package.

This is the lowest-level approach and therefore provides maximum control. It can still fail for some models (especially new ones), but you always have the option to dive inside the source code and try to figure out why.


The good news about Llama 2 is that we did the legwork and the conversion process works using any of these methods. The bad news is that it _failed to convert_ when it was released, and we had to do some fixing to support it. We briefly look at what happened in [the appendix](#appendix-converting-llama-2-the-hard-way) so you can get a taste of what to do when things go wrong.

### Important lessons learned

I've followed the conversion process for some recent models (Llama 2, Falcon, StarCoder), and I've applied what I learned to both `exporters` and the `transformers-to-coreml` Space. This is a summary of some takeaways:

- If you have to use `coremltools`, use the latest version: `7.0b1`. Despite technically being a beta, I've been using it for weeks and it's really good: stable, includes a lot of fixes, supports PyTorch 2, and has new features like advanced quantization tools.
- `exporters` no longer applies a softmax to outputs when converting text generation tasks. We realized this was necessary for some generation algorithms.
- `exporters` now defaults to using fixed sequence lengths for text models. Core ML has a way to specify "flexible shapes", such that your input sequence may have any length between 1 and, say, 4096 tokens. We discovered that flexible inputs only run on CPU, but not on GPU or the Neural Engine. More investigation coming soon!

We'll keep adding best practices to our tools so you don't have to discover the same issues again.

## Optimization

There's no point in converting models if they don't run fast on your target hardware and respect system resources. The models mentioned in this post are pretty big for local use, and we are consciously using them to stretch the limits of what's possible with current technology and understand where the bottlenecks are.

There are a few key optimization areas we've identified. They are a very important topic for us and the subject of current and upcoming work. Some of them include:

- Cache attention keys and values from previous generations, just like the transformers models do in the PyTorch implementation. The computation of attention scores needs to run on the whole sequence generated so far, but all the past key-value pairs were already computed in previous runs. We are currently _not_ using any caching mechanism for Core ML models, but are planning to do so!
- Use discrete shapes instead of a small fixed sequence length. The main reason not to use flexible shapes is that they are not compatible with the GPU or the Neural Engine. A secondary reason is that generation would become slower as the sequence length grows, because of the absence of caching as mentioned above. Using a discrete set of fixed shapes, coupled with caching key-value pairs should allow for larger context sizes and a more natural chat experience.
- Quantization techniques. We've already explored them in the context of Stable Diffusion models, and are really excited about the options they'd bring. For example, [6-bit palettization](https://huggingface.co/blog/fast-diffusers-coreml) decreases model size and is efficient with resources. [Mixed-bit quantization](https://huggingface.co/blog/stable-diffusion-xl-coreml), a new technique, can achieve 4-bit quantization (on average) with low impact on model quality. We are planning to work on these topics for language models too!

For production applications, consider iterating with smaller models, especially during development, and then apply optimization techniques to select the smallest model you can afford for your use case.

## `swift-transformers`

[`swift-transformers`](https://github.com/huggingface/swift-transformers) is an in-progress Swift package that aims to provide a transformers-like API to Swift developers. Let's see what it has and what's missing.

### Tokenizers

Tokenization solves two complementary tasks: adapt text input to the tensor format used by the model and convert results from the model back to text. The process is nuanced, for example:

- Do we use words, characters, groups of characters or bytes?
- How should we deal with lowercase vs uppercase letters? Should we even deal with the difference?
- Should we remove repeated characters, such as spaces, or are they important?
- How do we deal with words that are not in the model's vocabulary?

There are a few general tokenization algorithms, and a lot of different normalization and pre-processing steps that are crucial to using the model effectively. The transformers library made the decision to abstract all those operations in the same library (`tokenizers`), and represent the decisions as configuration files that are stored in the Hub alongside the model. For example, this is an excerpt from the configuration of the Llama 2 tokenizer that describes _just the normalization step_:

```
  "normalizer": {
    "type": "Sequence",
    "normalizers": [
      {
        "type": "Prepend",
        "prepend": "▁"
      },
      {
        "type": "Replace",
        "pattern": {
          "String": " "
        },
        "content": "▁"
      }
    ]
  },
```

It reads like this: normalization is a sequence of operations applied in order. First, we `Prepend` character `_` to the input string. Then we replace all spaces with `_`. There's a huge list of potential operations, they can be applied to regular expression matches, and they have to be performed in a very specific order. The code in the `tokenizers` library takes care of all these details for all the models in the Hub.

In contrast, projects that use language models in other domains, such as Swift apps, usually resort to hardcoding these decisions as part of the app's source code. This is fine for a couple of models, but then it's difficult to replace a model with a different one, and it's easy to make mistakes.

What we are doing in `swift-transformers` is replicate those abstractions in Swift, so we write them once and everybody can use them in their apps. We are just getting started, so coverage is still small. Feel free to open issues in the repo or contribute your own!

Specifically, we currently support BPE (Byte-Pair Encoding) tokenizers, one of the three main families in use today. The GPT models, Falcon and Llama, all use this method. Support for Unigram and WordPiece tokenizers will come later. We haven't ported all the possible normalizers, pre-tokenizers and post-processors - just the ones we encountered during our conversions of Llama 2, Falcon and GPT models.

This is how to use the `Tokenizers` module in Swift:

```swift
import Tokenizers

func testTokenizer() async throws {
    let tokenizer = try await AutoTokenizer.from(pretrained: "pcuenq/Llama-2-7b-chat-coreml")
    let inputIds = tokenizer("Today she took a train to the West")
    assert(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
}
```

However, you don't usually need to tokenize the input text yourself - the [`Generation` code](https://github.com/huggingface/swift-transformers/blob/17d4bfae3598482fc7ecf1a621aa77ab586d379a/Sources/Generation/Generation.swift#L82) will take care of it.

### Model and Hub wrappers

As explained above, `transformers` heavily use configuration files stored in the Hub. We prepared a simple `Hub` module to download configuration files from the Hub, which is used to instantiate the tokenizer and retrieve metadata about the model.

Regarding models, we created a simple `LanguageModel` type as a wrapper for a Core ML model, focusing on the text generation task. Using protocols, we can query any model with the same API.

To retrieve the appropriate metadata for the model you use, `swift-transformers` relies on a few custom metadata fields that must be added to the Core ML file when converting it. `swift-transformers` will use this information to download all the necessary configuration files from the Hub. These are the fields we use, as presented in Xcode's model preview:

![Screenshot: Core ML model metadata fields](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/swift-transformers/coreml-model-metadata.png)

`exporters` and `transformers-to-coreml` will automatically add these fields for you. Please, make sure you add them yourself if you use `coremltools` manually.

### Generation Algorithms

Language models are trained to predict a probability distribution of the next token that may appear as a continuation to an input sequence. In order to compose a response, we need to call the model multiple times until it produces a special _termination_ token, or we reach the length we desire. There are many ways to decide what's the next best token to use. We currently support two of them:

- Greedy decoding. This is the obvious algorithm: select the token with the highest probability, append it to the sequence, and repeat. This will always produce the same result for the same input sequence.
- top-k sampling. Select the `top-k` (where `k` is a parameter) most probable tokens, and then randomly _sample_ from them using parameters such as `temperature`, which will increase variability at the expense of potentially causing the model to go on tangents and lose track of the content.

Additional methods such as "nucleus sampling" will come later. We recommend [this blog post](https://huggingface.co/blog/how-to-generate) (updated recently) for an excellent overview of generation methods and how they work. Sophisticated methods such as [assisted generation](https://huggingface.co/blog/assisted-generation) can also be very useful for optimization!

### Supported Models

So far, we've tested `swift-transformers` with a handful of models to validate the main design decisions. We are looking forward to trying many more!

- Llama 2.
- Falcon.
- StarCoder models, based on a variant of the GPT architecture.
- GPT family, including GPT2, distilgpt, GPT-NeoX, GPT-J.

## `swift-chat`

`swift-chat` is a simple demo app built on `swift-transformers`. Its main purpose is to show how to use `swift-transformers` in your code, but it can also be used as a model tester tool.

![Swift Chat UI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/swift-transformers/swift-chat-ui.png)

To use it, download a Core ML model from the Hub or create your own, and select it from the UI. All the relevant model configuration files will be downloaded from the Hub, using the metadata information to identify what model type this is.

The first time you load a new model, it will take some time to prepare it. In this phase, the CoreML framework will compile the model and decide what compute devices to run it on, based on your machine specs and the model's structure. This information is cached and reused in future runs.

The app is intentionally simple to make it readable and concise. It also lacks a few features, primarily because of the current limitations in model context size. For example, it does not have any provision for "system prompts", which are [useful for specifying the behaviour of your language model](https://huggingface.co/blog/llama2#how-to-prompt-llama-2) and even its personality.

## Missing Parts / Coming Next

As stated, we are just getting started! Our upcoming priorities include:

- Encoder-decoder models such as T5 and Flan.
- More tokenizers: support for Unigram and WordPiece.
- Additional generation algorithms.
- Support key-value caching for optimization.
- Use discrete sequence shapes for conversion. Together with key-value caching this will allow for larger contexts.

Let us know what you think we should work on next, or head over to the repos for [Good First Issues](https://github.com/huggingface/swift-transformers/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to try your hand on!

## Conclusion

We introduced a set of tools to help Swift developers incorporate language models in their apps. I can't wait to see what you create with them, and I look forward to improving them with the community's help! Don't hesitate to get in touch :)

### _Appendix: Converting Llama 2 the Hard Way_

You can safely ignore this section unless you've experienced Core ML conversion issues and are ready to fight :)

In my experience, there are two frequent reasons why PyTorch models fail to convert to Core ML using `coremltools`:

- Unsupported PyTorch operations or operation variants

PyTorch has _a lot_ of operations, and all of them have to be mapped to an intermediate representation ([MIL](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html), for _Model Intermediate Language_), which in turn is converted to native Core ML instructions. The set of PyTorch operations is not static, so new ones have to be added to `coremltools` too. In addition, some operations are really complex and can work on exotic combinations of their arguments. An example of a recently-added, very complex op, was _scaled dot-product attention_, introduced in PyTorch 2. An example of a partially supported op is `einsum`: not all possible equations are translated to MIL.

- Edge cases and type mismatches

Even for supported PyTorch operations, it's very difficult to ensure that the translation process works on all possible inputs across all the different input types. Keep in mind that a single PyTorch op can have multiple backend implementations for different devices (cpu, CUDA), input types (integer, float), or precision (float16, float32). The product of all combinations is staggering, and sometimes the way a model uses PyTorch code triggers a translation path that may have not been considered or tested.

This is what happened when I first tried to convert Llama 2 using `coremltools`:

![Llama 2 conversion error](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/swift-transformers/llama-conversion-error.png)

By comparing different versions of transformers, I could see the problem started happening when [this line of code](https://github.com/huggingface/transformers/blob/d114a6b71f243054db333dc5a3f55816161eb7ea/src/transformers/models/llama/modeling_llama.py#L52C5-L52C6) was introduced. It's part of a recent `transformers` refactor to better deal with causal masks in _all_ models that use them, so this would be a big problem for other models, not just Llama.

What the error screenshot is telling us is that there's a type mismatch trying to fill the mask tensor. It comes from the `0` in the line: it's interpreted as an `int`, but the tensor to be filled contains `floats`, and using different types was rejected by the translation process. In this particular case, I came up with a [patch for `coremltools`](https://github.com/apple/coremltools/pull/1915), but fortunately this is rarely necessary. In many cases, you can patch your code (a `0.0` in a local copy of `transformers` would have worked), or create a "special operation" to deal with the exceptional case. Our `exporters` library has very good support for custom, special operations. See [this example](https://github.com/huggingface/exporters/blob/f134e5ceca05409ea8abcecc3df1c39b53d911fe/src/exporters/coreml/models.py#L139C9-L139C18) for a missing `einsum` equation, or [this one](https://github.com/huggingface/exporters/blob/f134e5ceca05409ea8abcecc3df1c39b53d911fe/src/exporters/coreml/models.py#L208C9-L208C18) for a workaround to make `StarCoder` models work until a new version of `coremltools` is released.

Fortunately, `coremltools` coverage for new operations is good and the team reacts very fast.

## Resources

- [`swift-transformers`](https://github.com/huggingface/swift-transformers).
- [`swift-chat`](https://github.com/huggingface/swift-chat).
- [`exporters`](https://github.com/huggingface/exporters).
- [`transformers-to-coreml`](https://huggingface.co/spaces/coreml-projects/transformers-to-coreml).
- Some Core ML models for text generation:
  - [Llama-2-7b-chat-coreml](https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml)
  - [Falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct/tree/main/coreml)
