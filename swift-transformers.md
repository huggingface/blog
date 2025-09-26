---
title: "Swift Transformers Reaches 1.0 – and Looks to the Future"
thumbnail: /blog/assets/swift-transformers/swift-transformers-thumb.png
authors:
- user: pcuenq
- user: FL33TW00D-HF
- user: mattt
- user: reach-vb
---

# Swift Transformers Reaches 1.0 – and Looks to the Future

We released [`swift-transformers`](https://github.com/huggingface/swift-transformers) two years ago (!) with the goal to support Apple developers and help them integrate local LLMs in their apps. A lot has changed since then (MLX and chat templates did not exist!), and we’ve learned how the community is actually using the library.

We want to double down on the use cases that provide most benefits to the community, and lay out the foundations for the future. Spoiler alert: after this release, we’ll focus a lot on MLX and agentic use cases 🚀

## What is `swift-transformers`

`swift-transformers` is a Swift library that [aims to reduce the friction](https://huggingface.co/blog/swift-coreml-llm) for developers that want to work with local models on Apple Silicon platforms, including iPhones. It includes the missing pieces that are not provided by Core ML or MLX alone, but that are required to work with local inference. Namely, it provides the following components:

* `Tokenizers`. Preparing inputs for a language model is surprisingly complex. We've built a lot of experience with our `tokenizers` Python and Rust libraries, which are foundational to the AI ecosystem. We wanted to bring the same performant, ergonomic experience to Swift. The Swift version of `Tokenizers` should handle everything for you, including chat templates and agentic use!
* `Hub`. This is an interface to the [Hugging Face Hub](https://huggingface.co), where all open models are available. It allows you to download models from the Hub and cache them locally, and supports background resumable downloads, model updates, offline mode. It contains a subset of the functionality provided by the [Python](https://huggingface.co/docs/huggingface_hub/en/index) and [JavaScript](https://huggingface.co/docs/huggingface.js/en/hub/README) libraries, focused on the tasks that Apple developers need the most (i.e., uploads are not supported).  
* `Models` and `Generation`. These are wrappers for LLMs converted to the Core ML format. Converting them is out of the scope of the library (but [we have some guides](https://huggingface.co/blog/mistral-coreml)). Once they are converted, these modules make it easy to run inference with them.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/swift-transformers/mlx-vlm-examples-birds.jpg" alt="Test app from mlx-swift-examples, showing SmolVLM2 explaining actions in a video." style="width: 100%;"/>
  <figcaption>Test app from mlx-swift-examples, showing SmolVLM2 explaining actions in a video.</figcaption>
</figure>

## How is the community using it

Most of the time people use the `Tokenizers` or `Hub` modules, and frequently both. Some notable projects that rely on `swift-transformers` include:

* [`mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples), by Apple. It’s, in fact, not just a collection of examples, but a list of libraries you can use to run various types of models using MLX, including LLMs and VLMs (vision-language models). It’s kind of our `Models` and `Generation` libraries but for MLX instead of Core ML – and it supports many more model types like embedders or Stable Diffusion.  
* [WhisperKit](https://github.com/argmaxinc/WhisperKit/), by [argmax](https://www.argmaxinc.com). Open Source ASR (speech recognition) framework, super heavily optimized for Apple Silicon. It relies on our `Hub` and `Tokenizers` modules.  
* [FastVLM](https://github.com/apple/ml-fastvlm/tree/main/app), by Apple, and many other app demos, such as our own [SmolVLM2 native app](https://huggingface.co/blog/smolvlm2).

## What changes with v1.0

Version 1.0 signals stability in the package. Developers are building apps on swift-transformers, and this first major release recognizes those use cases and brings the version number in line with that reality. It also provides the foundation on which to iterate with the community to build the next set of features. These are some of our preferred updates:

* **`Tokenizers` and `Hub`** are now [first-citizen, top-level modules](https://github.com/huggingface/swift-transformers/pull/269). Before 1.0, you had to depend on and import the full package, whereas now you can just pick `Tokenizers`, for instance.
* Speaking of Jinja, we are super proud to announce that we have collaborated with [John Mai](https://huggingface.co/JohnMai) ([X](https://x.com/JohnMai_Dev)) to create the **next version of his excellent Swift Jinja library**.
John’s work has been crucial for the community: he single-handedly took on the task to provide a solid chat template library that could grow as templates became more and more complex. The new version is a couple orders of magnitude faster (no kidding), and [lives here as `swift-jinja`](https://github.com/huggingface/swift-jinja).
* To further reduce the load imposed on downstream users, we have **removed our example CLI targets and the `swift-argument-parser` dependency**, which in turn prevents version conflicts for projects that already use it.
* Thanks to contributions by Apple, we have [adopted **Modern Core ML APIs**](https://github.com/huggingface/swift-transformers/pull/257) with support for stateful models (for easier KV-caching) and expressive `MLTensor` APIs – this removes thousands of lines of custom tensor operations and math code.
* Lots of **additional cruft removed and [API surface reduced](https://github.com/huggingface/swift-transformers/pull/250)** to reduce cognitive load and iterate faster.
* **Tests** are [better, faster, stronger](https://github.com/huggingface/swift-transformers/pull/229).
* **Documentation comments** have been added to [public APIs](https://github.com/huggingface/swift-transformers/pull/268).
* **Swift 6** is [fully supported](https://github.com/huggingface/swift-transformers/pull/264).

Version 1.0 comes with breaking API changes. However, we don’t expect major problems if you are a user of `Tokenizers` or `Hub`. If you use the Core ML components of the library, please [get in touch](https://github.com/huggingface/swift-transformers/issues/new) so we can support you during transition. We’ll prepare a migration guide and add it to the documentation.

## Usage Examples

Here's how to use `Tokenizers` to format tool calling input for an LLM:

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(pretrained: "mlx-community/Qwen2.5-7B-Instruct-4bit")

let weatherTool = [
    "type": "function",
    "function": [
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": [
            "type": "object",
            "properties": ["location": ["type": "string", "description": "City and state"]],
            "required": ["location"]
        ]
    ]
]

let tokens = try tokenizer.applyChatTemplate(
    messages: [["role": "user", "content": "What's the weather in Paris?"]],
    tools: [weatherTool]
)
```

For additional examples, please check [this section in the README](https://github.com/huggingface/swift-transformers?#examples) and the [Examples folder](https://github.com/huggingface/swift-transformers/tree/main/Examples).

## What comes next

Honestly, we don’t know. We do know that we are super interested in exploring MLX, because that’s usually the current go-to approach for developers getting started with ML in native apps, and we want to help make the experience as seamless as possible. We are thinking along the lines of better integration with `mlx-swift-examples` for LLMs and VLMs, potentially through pre-processing and post-processing operations that developers encounter frequently.

We are also extremely excited about agentic use in general and MCP in particular. We think that exposure of system resources to local workflows would be 🚀

If you want to follow along in this journey or want to share your ideas, please contact us through our social networks or [the repo](https://github.com/huggingface/swift-transformers/issues/new).

## We couldn’t have done this without you 🫵

We are immensely grateful to all the contributors and users of the library for your help and feedback. We love you all, and can't wait to continue working with you to shape the future of on-device generation! ❤️