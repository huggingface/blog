---
title: "Transformers.js v4 Preview: Now Available on NPM!"
thumbnail: /blog/assets/transformersjs-v4/thumbnail.png
authors:
  - user: Xenova
  - user: nico-martin
---

# Transformers.js v4 Preview: Now Available on NPM!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformersjs-v4/thumbnail-wide.png" alt="Overview" width="100%">

We're excited to announce that Transformers.js v4 (preview) is now available on NPM! After nearly a year of development (starting in March 2025), we're finally ready for you to test it out. Previously, users had to install v4 directly from source via GitHub, but now it's as simple as running a single command!

```sh
npm i @huggingface/transformers@next
```

We'll continue publishing v4 releases under the `next` tag on NPM until the full release, so expect regular updates!

## Performance & Runtime Improvements

The biggest change is undoubtedly the adoption of a new WebGPU Runtime, completely rewritten in C++. We've worked closely with the ONNX Runtime team to thoroughly test this runtime across our ~200 supported model architectures, as well as many new v4-exclusive architectures.

In addition to better operator support (for performance, accuracy, and coverage), this new WebGPU runtime allows the same library (and code) to be used across a wide variety of JavaScript environments, including browsers, server-side runtimes, and desktop applications. That's right, you can now run WebGPU-accelerated models directly in Node, Bun, and Deno!

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformersjs-v4/webgpu.png" alt="WebGPU Overview" width="100%">

We've proven that it's possible to run state-of-the-art AI models 100% locally in the browser, and now we're focused on performance: making these models run as fast as possible, even in resource-constrained environments. This required completely rethinking our export strategy, especially for large language models. We achieve this by re-implementing new models operation by operation, leveraging specialized ONNX Runtime [Contrib Operators](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md) like [com.microsoft.GroupQueryAttention](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention), [com.microsoft.MatMulNBits](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits), and [com.microsoft.QMoE](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QMoE) to maximize performance.

For example, by utilizing the [com.microsoft.MultiHeadAttention](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MultiHeadAttention) operator, we were able to achieve a ~4x speedup for BERT-based embedding models.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformersjs-v4/speedups.png" alt="Optimized ONNX Exports" width="100%">

Finally, this update enables full offline support by caching WASM files locally in the browser, allowing users to run Transformers.js applications without an internet connection after the initial download.

## Repository Restructuring

Developing a new major version gave us the opportunity to invest in the codebase and tackle long-overdue refactoring efforts.

### PNPM Workspaces

Until now, the GitHub repository served as our npm package. This worked well as long as the repository only exposed a single library. However, looking to the future, we saw the need for various sub-packages that depend heavily on the Transformers.js core while addressing different use cases, like library-specific implementations, or smaller utilities that most users don't need but are essential for some.

That's why we converted the repository to a monorepo using pnpm workspaces. This allows us to ship smaller packages that depend on `@huggingface/transformers` without the overhead of maintaining separate repositories.

### Modular Class Structure

Another major refactoring effort targeted the ever-growing models.js file. In v3, all available models were defined in a single file spanning over 8,000 lines, becoming increasingly difficult to maintain. For v4, we split this into smaller, focused modules with a clear distinction between utility functions, core logic, and model-specific implementations. This new structure improves readability and makes it much easier to add new models. Developers can now focus on model-specific logic without navigating through thousands of lines of unrelated code.

### Examples Repository

In v3, many Transformers.js example projects lived directly in the main repository. For v4, we've moved them to a [dedicated repository](https://github.com/huggingface/transformers.js-examples), allowing us to maintain a cleaner codebase focused on the core library. This also makes it easier for users to find and contribute to examples without sifting through the main repository.

### Prettier

We updated the Prettier configuration and reformatted all files in the repository. This ensures consistent formatting throughout the codebase, with all future PRs automatically following the same style. No more debates about formatting... Prettier handles it all, keeping the code clean and readable for everyone.

## New Models and Architectures

Thanks to our new export strategy and ONNX Runtime's expanding support for custom operators, we've been able to add many new models and architectures to Transformers.js v4. These include popular models like GPT-OSS, Chatterbox, GraniteMoeHybrid, LFM2-MoE, HunYuanDenseV1, Apertus, Olmo3, FalconH1, and Youtu-LLM. Many of these required us to implement support for advanced architectural patterns, including Mamba (state-space models), Multi-head Latent Attention (MLA), and Mixture of Experts (MoE). Perhaps most importantly, these models are all compatible with WebGPU, allowing users to run them directly in the browser or server-side JavaScript environments with hardware acceleration. Stay tuned for some exciting demos showcasing these new models in action!

## New Build System

We've migrated our build system from Webpack to esbuild, and the results have been incredible. Build times dropped from 2 seconds to just 200 milliseconds, a 10x improvement that makes development iteration significantly faster. Speed isn't the only benefit, though: bundle sizes also decreased by an average of 10% across all builds. The most notable improvement is in transformers.web.js, our default export, which is now 53% smaller, meaning faster downloads and quicker startup times for users.

## Standalone Tokenizers.js Library

A frequent request from users was to extract the tokenization logic into a separate library, and with v4, that's exactly what we've done. [@huggingface/tokenizers](https://www.npmjs.com/package/@huggingface/tokenizers) is a complete refactor of the tokenization logic, designed to work seamlessly across browsers and server-side runtimes. At just 8.8kB (gzipped) with zero dependencies, it's incredibly lightweight while remaining fully type-safe.

<details>
<summary>See example code</summary>

```javascript
import { Tokenizer } from "@huggingface/tokenizers";

// Load from Hugging Face Hub
const modelId = "HuggingFaceTB/SmolLM3-3B";
const tokenizerJson = await fetch(
  `https://huggingface.co/${modelId}/resolve/main/tokenizer.json`
).then(res => res.json());

const tokenizerConfig = await fetch(
  `https://huggingface.co/${modelId}/resolve/main/tokenizer_config.json`
).then(res => res.json());

// Create tokenizer
const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);

// Tokenize text
const tokens = tokenizer.tokenize("Hello World");
// ['Hello', 'ĠWorld']

const encoded = tokenizer.encode("Hello World");
// { ids: [9906, 4435], tokens: ['Hello', 'ĠWorld'], ... }
```

</details>

This separation keeps the core of Transformers.js focused and lean while offering a versatile, standalone tool that any WebML project can use independently.

## Miscellaneous Improvements

We've made several quality-of-life improvements across the library. The type system has been enhanced with dynamic pipeline types that adapt based on inputs, providing better developer experience and type safety.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformersjs-v4/types.png" alt="Type Improvements" width="100%">

Logging has been improved to give users more control and clearer feedback during model execution. Additionally, we've added support for larger models exceeding 8B parameters. In our tests, we've been able to run GPT-OSS 20B (q4f16) at ~60 tokens per second on an M4 Pro Max.

## Acknowledgements

We want to extend our heartfelt thanks to everyone who contributed to this major release, especially the ONNX Runtime team for their incredible work on the new WebGPU runtime and their support throughout development, as well as all external contributors and early testers.
