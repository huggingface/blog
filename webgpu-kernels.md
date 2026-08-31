---
title: "Introducing @huggingface/kernels: 200+ WebGPU Kernels for Browser AI"
thumbnail: /blog/assets/webgpu-kernels/thumbnail.png
authors:
  - user: nico-martin
  - user: Xenova
---

# Introducing `@huggingface/kernels`: 200+ WebGPU Kernels for Browser AI

One of our biggest goals on the WebAI team at Hugging Face is to make browser inference as fast and as user-friendly as possible. Getting there is a multi-layer effort: models need browser-friendly representations, runtimes need to build efficient execution plans, and the individual GPU operations at the bottom of the stack need to make the most of many different devices and browser implementations.

Today, we are releasing the first layer of that effort: [`@huggingface/kernels`](https://www.npmjs.com/package/@huggingface/kernels), a library for loading and running optimized WebGPU kernels from the Hugging Face Hub, together with an initial collection of **207 kernels** at [huggingface.co/webgpu-kernels](https://huggingface.co/webgpu-kernels).

The collection covers operations used across a wide variety of machine learning architectures and workloads. More importantly, each kernel is published as a complete, versioned package: its interface, shader templates, correctness cases, benchmark cases, provenance, and usage instructions all live together on the Hub.

We are also launching [WebGPU Kernels Fleet](https://webgpu-kernels-fleet.hf.space/), an in-browser test suite that lets you benchmark the kernels on your own GPU. Beyond the numbers for your own machine, Fleet gives the community a way to contribute performance and correctness evidence from devices we could never cover in a conventional test lab. With your consent, every run adds private evidence that can help us find failures, improve kernel variants, and make better optimization decisions across real-world hardware.

## TL;DR

- **207 WebGPU kernels**, published as individual repositories in the [`webgpu-kernels`](https://huggingface.co/webgpu-kernels) organization.
- **A JavaScript loader**, `@huggingface/kernels`, which downloads, prepares, and runs kernels directly from the Hub.
- **Explicit contracts and reproducible evidence** for every kernel, including manifests, correctness tests, benchmark cases, and WGSL shader templates.
- **WebGPU Kernels Fleet**, a browser-based campaign that crowdsources correctness and performance evidence across real-world GPUs to help us improve kernels and their variants.

## Why start with kernels?

A model running in the browser eventually becomes a sequence of GPU operations: matrix multiplications, normalizations, convolutions, attention primitives, quantization operations, data-layout transformations, and many more. WebGPU makes these operations available across modern browsers through a portable API, while WGSL provides a common language for the shaders that execute them.

Portability, however, does not automatically mean performance. Two shaders can implement the same operation and produce the same output while behaving very differently on a particular GPU. Workgroup sizes, memory access patterns, vectorization, data types, and fusion strategies can all affect performance. The best choice can also change with the input shape, device, browser, and available WebGPU features.

This is why kernels form a foundational layer of fast browser inference. Higher-level runtimes can only be as efficient as the operations they dispatch. By making those operations individually discoverable, testable, benchmarkable, and versioned, we can improve the foundation independently while keeping a stable contract for the layers above it.

## A kernel repository, not just a shader

Each kernel in the collection has its own repository and kernel card. The card documents the operation's semantics, inputs, outputs, attributes, supported data types, source files, and a ready-to-run `@huggingface/kernels` example.

For example, [`ai.onnx.Add`](https://huggingface.co/webgpu-kernels/ai.onnx.Add) implements elementwise addition with multidirectional broadcasting. It is one of the simplest operations in a neural network, used everywhere from residual connections to adding a bias. Its card documents the two inputs, the broadcasted output shape, supported data types, and the variants available for different shapes and devices.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/webgpu-kernels/ai-onnx-add.png" alt="Files in the ai.onnx.Add WebGPU kernel repository" width="90%">
  <figcaption>The <code>ai.onnx.Add</code> repository packages its manifest, correctness and benchmark cases, and WGSL shader templates together.</figcaption>
</figure>

Behind the card, the repository contains the artifacts needed to understand and evaluate the implementation:

- **`manifest.json`** is the source of truth for the operation contract. It defines inputs, outputs, attributes, type constraints, and shape derivation rules.
- **`metadata.json`** records the kernel identifier, digests, and provenance.
- **`test.json`** contains correctness cases, so an implementation can be checked against expected behavior.
- **`bench.json`** contains benchmark and tuning cases that represent the workloads used to evaluate the kernel.
- **`*.wgsl.jinja`** files contain the parameterized WGSL implementations used to produce shaders for a particular request and device.

This structure turns a shader into a reusable software artifact. The interface is inspectable without reading WGSL, correctness and performance cases travel with the implementation, and published versions can be loaded explicitly rather than depending on an unversioned file URL.

## Loading a kernel from the Hub

Install the package from npm:

```bash
npm install @huggingface/kernels
```

> [!NOTE]
> Running these kernels requires a browser with [WebGPU support](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API). WebGPU availability depends on the browser, operating system, GPU, and driver. You can check for it in JavaScript with `"gpu" in navigator`.

`@huggingface/kernels` provides the bridge between a kernel repository and your application. Call `getKernel` with a Hub repository ID and a contract version, then invoke the returned function with typed input data and tensor shapes. Here is a small bias-add example:

```js
import { getKernel } from "@huggingface/kernels";

const add = await getKernel("webgpu-kernels/ai.onnx.Add", { version: 1 });

const { c } = await add({
  a: {
    data: new Float32Array([1, 2, 3, 4, 5, 6]),
    shape: [2, 3],
  },
  b: {
    data: new Float32Array([10, 20, 30]),
    shape: [3],
  },
});
```

The second input is broadcast across the first dimension, producing an output with shape `[2, 3]`. The loader derives that output shape and logical data type from the manifest contract and the inputs, then allocates `c` automatically.

Addition on six floats is deliberately the smallest possible demo. At this size, the GPU round trip costs far more than the math. The point is the call pattern: it stays exactly the same for the heavyweight operations where optimized kernels actually pay off, such as matrix multiplication (`ai.onnx.MatMul`). Only the repository ID and the inputs change.

Even this elementary operation illustrates why kernels need variants. Equal-shape addition can use a direct vectorized path, while broadcasted inputs need different indexing logic. The published Add kernel includes variants for equal shapes, vectorized broadcasting, scalar processing, and general broadcasting. The runtime can select an implementation that fits the current call and device without changing the application-facing API.

The `version: 1` option selects version 1 of the published **kernel contract**. It is separate from an ONNX opset, an operator's `since_version`, or a model revision. Keeping those concepts separate lets applications depend on a stable JavaScript-facing contract while kernel implementations evolve behind it.

## How fast are the kernels?

As a focused first comparison, we benchmarked published Add, MatMul, Softmax, and LayerNormalization kernels against the WebGPU execution provider in ONNX Runtime Web 1.29.0. The kernel manifests and WGSL templates came directly from pinned commits on each Hub repository's `v1` branch, and their published SHA-256 file digests were verified before execution. Both implementations ran on the same `GPUDevice` with inputs already uploaded to GPU buffers. Each measurement includes one operation, runtime submission, synchronization, and copying the output back to JavaScript; downloading and compiling the kernels happens before measurement.

The table reports the median of five independent runs, each with 5 warmup iterations followed by 30 measured iterations, on a 40-core Apple M3 Max GPU using Chrome 151 on macOS 26.6.2. We reversed the Hub/ONNX Runtime execution-order pattern between runs to reduce systematic ordering bias:

| Operation and workload | Selected kernel variant | Hub kernel | ONNX Runtime Web | Speedup |
| --- | --- | ---: | ---: | ---: |
| Add: `[4096, 4096] + [4096, 4096]` | `same_shape_vec4` | 8.50 ms | 18.75 ms | **2.21x** |
| Add: `[4096, 4096] + [4096]` | `broadcast_vec4` | 8.85 ms | 19.70 ms | **2.23x** |
| MatMul: `[2048, 2048] @ [2048, 2048]` | `rank2_notrans_f32_vec4_tiled_reg` | 8.70 ms | 10.80 ms | **1.24x** |
| Softmax: `[16384, 512]`, axis 1 | `online_wg_vec4` | 7.35 ms | 10.00 ms | **1.36x** |
| LayerNormalization: `[4096, 4096]` | `last_axis_bias_row_vec4` | 8.50 ms | 19.60 ms | **2.31x** |

All workloads used `float32`, and both implementations produced matching outputs within the expected numerical tolerances. For reproducibility, the pinned Hub revisions were `9663d5f` for Add, `de365a6` for MatMul, `a8e6c70` for Softmax, and `49b0523` for LayerNormalization.

These are public-API latency measurements, not isolated shader timings. They include differences in each runtime's command submission and GPU-to-CPU output path, and results on other GPUs and browsers will vary. Individual operation results should not be treated as a proxy for whole-model performance. Fleet is how we intend to build a broader picture across operations and devices.

## From one device to a fleet

Writing a correct kernel is only the beginning. The WebGPU ecosystem spans integrated and discrete GPUs, different vendors, different operating systems, and multiple browser GPU stacks. A kernel that performs well on one machine may compile slowly, select a poor variant, or expose a driver issue somewhere else. No fixed collection of development machines can represent that full range.

[WebGPU Kernels Fleet](https://webgpu-kernels-fleet.hf.space/) is how we crowdsource coverage for that long tail. It is a WebGPU campaign that runs entirely in the browser tab. After a device census, the Standard campaign sends a balanced set of correctness and benchmark packets to your GPU. Faster devices can continue into additional coverage, while slower devices stop at a safe boundary so the trial remains practical.

The benchmark results you see are useful, but the aggregate evidence is the core of Fleet. Each consenting run expands our view of how kernels and their variants behave on a particular combination of GPU, browser, driver, features, and limits. More runs give us more opportunities to identify device-specific failures, discover weak variants, tune selection rules, and validate fixes. In other words, joining the Fleet directly helps improve the kernels for everyone.

The campaign measures several complementary dimensions:

| Area | What it measures |
| --- | --- |
| **Bandwidth** | Sustained memory throughput |
| **Compute** | Floating-point throughput |
| **Quantized** | 8-bit dot-product throughput |
| **Latency** | Dispatch and compilation responsiveness |
| **Efficiency** | Real operation performance compared with measured hardware ceilings |
| **Stability** | Repeatability and thermal behavior |

These measurements provide context that a kernel benchmark alone cannot. For example, comparing operation performance with the device's measured bandwidth and compute ceilings helps distinguish a kernel limitation from a hardware limit. Correctness packets also help identify combinations of shader, browser, and GPU that need further investigation. Across many contributed devices, this evidence becomes a feedback loop for improving implementations and choosing better variants.

## Building a shared foundation for WebAI

The initial 207 kernels are a starting point, not the end state. Publishing kernels independently on the Hub gives us a common place to inspect contracts, compare implementations, reproduce correctness checks, and improve performance without embedding every shader directly into every runtime.

The collection is also part of the Hub's broader kernel ecosystem: on the [Kernels page](https://huggingface.co/kernels?platform=webgpu&sort=trending), the WebGPU kernels sit alongside kernels for CUDA, ROCm, Metal, and other platforms, and can be filtered, sorted, and explored like any other artifact on the Hub.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/webgpu-kernels/kernels.png" alt="The Hub Kernels page filtered to the WebGPU platform, listing the 207 published kernels" width="90%">
  <figcaption>All 207 WebGPU kernels on the Hub's <a href="https://huggingface.co/kernels?platform=webgpu&sort=trending">Kernels page</a>, filtered by platform.</figcaption>
</figure>

The pieces reinforce one another:

1. Kernel repositories define transparent, versioned operation contracts.
2. `@huggingface/kernels` makes those operations straightforward to load from JavaScript.
3. Fleet crowdsources real-world evidence across a much broader range of devices than a conventional benchmark lab can cover.
4. Every contributed run can reveal failures, guide tuning, improve variant selection, and help validate future kernel versions.

This is the low-level foundation for the next steps in our browser inference stack. We are excited to connect these kernels to higher-level model tooling, continue expanding operation coverage, and make fast local inference easier to use across the WebAI ecosystem.

Explore the [WebGPU kernel collection](https://huggingface.co/webgpu-kernels), try [`@huggingface/kernels`](https://www.npmjs.com/package/@huggingface/kernels), and [join the Fleet](https://webgpu-kernels-fleet.hf.space/) to contribute evidence from your device and help us make the kernels better for everyone.
