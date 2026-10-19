---
title: "AnyLanguageModel 1.0: One Year of Local Models on Apple Platforms"
thumbnail: /blog/assets/anylanguagemodel-1-0/banner.png
authors:
- user: mattt
---

Last November, we introduced [AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel) on this blog:
a drop-in replacement for Apple's Foundation Models framework
that lets you choose the model and provider your app uses.
We heard from app developers who were curious about local models
but weren't always sure how to get started.
They also wanted to avoid the added complexity
of supporting multiple inference providers.
AnyLanguageModel grew out of that feedback.

Local and hosted inference each have their strengths,
and both have a role in building this new generation of intelligent apps.
A shared API makes it easier to choose the right model for each task.

A year and dozens of releases later,
[AnyLanguageModel 1.0](https://github.com/huggingface/AnyLanguageModel/releases/tag/1.0.0) is out,
along with [swift-huggingface 1.0](https://github.com/huggingface/swift-huggingface/releases/tag/1.0.0)
and [swift-xet 1.0](https://github.com/huggingface/swift-xet/releases/tag/1.0.0).

Here's what changed, and what's next:

## A drop-in compatible API

The core idea of AnyLanguageModel remains the same:
use the Foundation Models API with the model that fits your app.
To get started, change your `import` statement:

```diff
- import FoundationModels
+ import AnyLanguageModel
```

The same session code then works with the system model,
with open-weight models through llama.cpp, MLX, and Core ML,
and with the hosted providers below.

Over the past year, we've added several major features:

- **Tool calling** works with every provider.
- **Guided generation** works with every provider, too,
  including nested `@Generable` types and arrays of them.
- **Image support** lets you attach images to prompts
  using Foundation Models 27's syntax,
  including when you use the system model on iOS, macOS, and visionOS 27.

On OS 27, AnyLanguageModel also works with
any model that conforms to Foundation Models' `LanguageModel` protocol,
including [Core AI](https://github.com/apple/coreai-models) models.

Alongside AnyLanguageModel, we've built out
[swift-huggingface](https://github.com/huggingface/swift-huggingface)
and [swift-xet](https://github.com/huggingface/swift-xet)
to bring accelerated model downloads from the Hub to Swift apps.

## Hosted models, too

Support for hosted models expanded this year, too.
AnyLanguageModel now supports:

- OpenAI's Chat Completions and Responses APIs
- [Open Responses](https://www.openresponses.org),
  for any provider with a compatible endpoint
- Anthropic's Messages API, with adaptive thinking and effort options
- Google's Gemini API
- Ollama
- Private Cloud Compute, on OS 27
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index),
  and other OpenAI-compatible endpoints such as OpenRouter,
  through a custom base URL

With Inference Providers,
you can use open-weight models that are too big to run on the device
with the same session code:

```swift
let model = OpenAILanguageModel(
    baseURL: URL(string: "https://router.huggingface.co/v1/")!,
    apiKey: ProcessInfo.processInfo.environment["HF_TOKEN"]!,
    model: "openai/gpt-oss-120b",
    apiVariant: .chatCompletions
)
```

Hosted providers get the same features as local models:
guided generation, tool calling with streaming,
and token usage on responses.

Reasoning is now part of the transcript too.
When you switch providers,
reasoning blocks are automatically omitted from requests if the new provider can't replay them.

API keys come from a closure instead of a string,
so your app can fetch short-lived tokens from your own server.
The [README](https://github.com/huggingface/AnyLanguageModel#api-credentials-and-security)
explains when to let people bring their own keys
and when to route requests through a proxy.

AnyLanguageModel also runs on Linux,
so the same code can work on a server.

## Open-weight models got better

While we've made local models easier to use,
the models themselves have become more capable,
making the small ones a good place to start.
Here are some of the most downloaded models on the Hub right now:

- [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) from Google,
  in GGUF, about 3.4 GB at `Q4_0` with quantization-aware training.
- [LFM2.5 2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) from Liquid AI,
  in GGUF, about 1.7 GB at `Q4_K_M`.
- [Qwen3.8 27B for MLX](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) from Alibaba,
  about 16 GB at 4 bits.

The first two fit on an iPhone,
and Qwen3.8 27B runs on a Mac with 32 GB of memory or more.
Selecting the MLX model takes one line:

```swift
let model = MLXLanguageModel(modelId: "mlx-community/Qwen3.8-27B-4bit")
```

And if you have four maxed-out Mac Studios handy,
[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) from Z.ai takes about 1.5 TB at full precision.
For the rest of us, it's available through Inference Providers.

## Upgrading to 1.0

```swift
dependencies: [
    .package(
        url: "https://github.com/huggingface/AnyLanguageModel",
        from: "1.0.0",
        traits: ["MLX"]
    )
]
```

If you're upgrading an existing app, see the migration guidance in the
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel/releases/tag/1.0.0),
[swift-huggingface](https://github.com/huggingface/swift-huggingface/releases/tag/1.0.0),
and [swift-xet](https://github.com/huggingface/swift-xet/releases/tag/1.0.0) release notes
for help updating your code.

Community feedback and contributions have shaped several of our major features.
Thank you to [everyone who contributed](https://github.com/huggingface/AnyLanguageModel/graphs/contributors)
for helping make AnyLanguageModel more useful to app developers.

## What's next

Language models have proven to be a powerful tool
for building intelligence into products.
They're also one of many kinds of models developers can put to work.
We're excited to bring the same approach to more capabilities:
familiar Swift APIs with a choice of models running on the device.
Coming soon: models that decide, and models that listen.

## Links

- [AnyLanguageModel on GitHub](https://github.com/huggingface/AnyLanguageModel)
- [swift-huggingface on GitHub](https://github.com/huggingface/swift-huggingface)
- [swift-xet on GitHub](https://github.com/huggingface/swift-xet)
- [Introducing AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel)
- [Introducing swift-huggingface](https://huggingface.co/blog/swift-huggingface)
