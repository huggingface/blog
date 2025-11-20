---
title: "The Transformers Library: standardizing model definitions"
thumbnail: /blog/assets/transformers-model-definition/transformers-thumbnail.png
authors:
  - user: lysandre
  - user: ArthurZ
  - user: cyrilvallez
  - user: reach-vb
---

# Transformers v5

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_v5/transformers-thumbnail.png" alt="Transformers standardizing model definitions">

***Transformers v5: Simple model definitions powering the AI ecosystem***

## Intro

Transformers' version v4.0.0rc-1, the initial release candidate for version 4, was released on November 19th, 2020\. Five years later, we release v5.0.0rc-1.

Today, with the launch of v5, Transformers is downloaded over **3 million times per day via pip** \- up from just **20,000/day** in v4. Altogether, it has now surpassed **1.2 billion installs**.

The ecosystem has expanded from **40 model architectures in v4** to **over 400 today**, and the community has contributed **more than 750,000 model checkpoints on the Hub** compatible with Transformers, up from roughly **1,000** at the time of v4.

This growth is linked to the growth of the field and the now mainstream access of AI, no doubt; as a leading model-definition library in the ecosystem, we need to continuously evolve and adapt the library to continue being relevant. Reinvention is key for longevity in AI.

We’re lucky to be working with a great number of libraries and apps working with transformers, in no specific order: llama.cpp, MLX, onnxruntime, Jan, LMStudio, vLLM, SGLang, Unsloth, LlamaFactory , dLLM, MaxText, TensorRT, Argmax, among many other friends.

For v5, we wanted to work on several notable aspects: simplicity, training, inference, and production. We detail the work that went into them in this post.

## Simplicity

The first focus of the team was on simplicity. Working on transformers, we see the code as the product. We want our model integrations to be clean so that other libraries in the ecosystem may depend on our model definitions and understand what’s really happening under the hood,how models differ from each other, and what are the key features of each new model. Simplicity results in wider standardization, generality, and wider support.

### Model Additions

Transformers, at the core, remains a model architecture toolkit: we aim to have all recent architectures, and to act as the “source of truth” of such model definitions in the ecosystem. We’ve been adding between 1 and 3 new architectures to the toolkit every week over the past 5 years, as can be seen in the timeline below:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_v5/transformers_model_timeline.png" alt="Transformers standardizing model definitions">

[*https://huggingface.co/spaces/yonigozlan/Transformers-Timeline*](https://huggingface.co/spaces/yonigozlan/Transformers-Timeline)

We’ve worked on improving that model-addition process.

#### Modular Approach 

Over the past year, we’ve heavily pushed our modular design as a significant step forward. This allows for easier maintenance, faster integration, and better collaboration across the community. 

We give a deeper overview in our [*Maintain the Unmaintainable*](https://huggingface.co/spaces/transformers-community/Transformers-tenets) blog post, but this ensures long-term a much greater ease of model addition, and a lowered maintenance burden. The resulting lines of code to contribute, and review, therefore drops significantly:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_v5/modular_timeline.png" alt="Transformers standardizing model definitions">

#### Tooling for Model Conversion

We’re building tooling to help us identify which existing model architecture a new model resembles; this uses machine learning to find code similarities between independent modeling files. Going further, we aim to automate the conversion process by opening a draft PR for the model to be integrated into our transformers format. This reduces manual effort and ensures consistency.

### Code Reduction

#### Streamlining Modeling & Tokenization/Processing Files

We’ve significantly refactored the modeling and tokenization files; modeling files have been greatly improved thanks to the modular approach mentioned above, but also thanks to significant standardization across models. This process contributes to abstracting most of the tools that don’t make up a model, so that the modeling code only contains the relevant parts for a model’s forward/backward passes.

Alongside this work, we’re simplifying the tokenization and processing files: going forward, we’ll only focus on the `tokenizers` backend, removing the concept of “Fast” and “Slow” tokenizers***.*** 

We’ll be using `tokenizers` as our primary tokenization backend, same as we do for models with torch. We’ll offer alternatives for Sentencepiece-backed or MistralCommon-backed tokenizers, which will be non-default but will be supported. Image processors will now only exist with their fast variant, that depends on the `torchvision` backend.

Finally, we’re sunsetting our Flax/TensorFlow support in favor of focusing solely on `torch` going forward.

#### Abstractions and Simplifications

While we respect the “One model, one file” philosophy, we continue introducing some abstractions making the management of common helpers simpler. The prime example of this is the introduction of the `AttentionInterface`, which offers a centralized abstraction for attention methods. The \`eager\` method will remain in the modeling file; others, such as FA1/2/3, FlexAttention, or SDPA, are moved to the interface.

## Training

Training remains a big focus of the team as we head into v5: whereas previously we would focus heavily on fine-tuning rather than pre-training/full-training at scale, we’ve recently done significant work to improve our support for the latter as well.

### Pre-training at scale

Supporting pre-training meant reworking the initialization of our models, ensuring that they worked at scale with different parallelism paradigms, and shipping support for optimized kernels with both the forward, and backward passes optimized.

Going forward, we’re excited to have extended compatibility with torchtitan, megatron, nanotron, as well as any other pre-training tool that is interested in collaborating with us.

### Fine-tuning & Post-training

We continue collaborating closely with all fine-tuning tools in the Python ecosystem. We aim to continue providing model implementations compatible with Unsloth, Axolotl, LlamaFactory, TRL and others in the PyTorch ecosystem; but we are also working with tools such as MaxText, in the Jax ecosystem, to have good interoperability between their frameworks and `transformers`.

All fine-tuning and post-training tools should have the possibility to rely on transformers for model definitions; further enabling Agentic use-cases through OpenEnv or the Prime Environment Hub.

## Inference

We’re putting a significant focus on inference for v5, with several paradigm changes: the introduction of specialized kernels, cleaner defaults, new APIs, support for optimized inference engines.

Similarly to training, we’ve been putting some effort in packaging `kernels` so that they’re automatically used in case your hardware and software permits it. If you haven’t heard of `kernels` before, we recommend taking a look at this [doc](https://huggingface.co/docs/kernels/basic-usage).

Alongside this effort, we ship two new APIs dedicated to inference:

- We ship support for continuous batching and paged attention mechanisms. This has now been used internally for some time, and we’re working on finalizing the rough edges and writing usage guides.  
- We introduce `transformers serve` as the new transformers-specific serving system, which deploys an OpenAI API-compatible server.

We see this as a major step forward for use-cases such as evaluation, where a great number of inference requests are done simultaneously. We don’t aim to do specialized optimizations like the dedicated inference engines (vLLM, SGLang, TensorRT LLM). Instead, we aim to be perfectly inter-compatible with these, as detailed in the next section.

### Production & Local

Recently, we've been working hand in hand with the most popular inference engines for them to use `transformers` as a backend. The value added is significant: as soon as a model is added to `transformers`, it becomes available in these inference engines, *while taking advantage of the strengths each engine provides*: inference optimizations, specialized kernels, dynamic batching, etc.

We've also been working very closely with ONNXRuntime, [llama.cpp](https://github.com/ggml-org/llama.cpp) and [MLX](https://github.com/ml-explore/mlx) so that the implementations between `transformers` and these modeling libraries have great interoperability. For example, thanks to a significant community effort, it's now very easy to [load GGUF files in `transformers`](https://huggingface.co/docs/transformers/en/gguf) for further fine-tuning. Conversely, transformers models can be easily [converted to GGUF files](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) for use with llama.cpp.

The same is true for MLX, where the transformers' safetensors files are directly compatible with MLX's models.

Finally, we’re pushing the boundaries of local inference and are working hand-in-hand with the `executorch` team to get the transformers models to be available on-device. We’re expanding the coverage to multimodal models (vision, audio).

## Quantization

Quantization is quickly emerging as the standard for state-of-the-art model development. Many SOTA models are now released in low-precision formats such as 8-bit and 4-bit (e.g gpt-oss, Kimi-K2, Deepseek-r1), hardware is increasingly optimized for low-precision workloads, and the community is actively sharing high-quality quantized checkpoints. In v5, we're making quantization a central focus of Transformers support, ensuring full compatibility with all major features, and delivering a reliable framework for training and inference.

## Conclusion

The overarching theme of this version 5 release is “interoperability”. All refactors, performance improvements, and standardization are aligned with this theme. Train a model with Unsloth/Axolotl/LlamaFactory/MaxText, deploy it with vLLM/SGLang, and export it to llama.cpp/executorch/MLX to run locally\! 

Version 5 is undeniably an accomplishment of the past five years by a very large number of people in our community. We also see it as a promise, and as a beacon of the direction we want to go.

We took it as an opportunity to greatly clean up the toolkit and isolate what mattered; we now have a clean slate on top of which it is much simpler to build. Thanks to the many changes of the community and the team, improvements in performance, usability, and readability, will be much simpler to ship.

Now that v5.0.0's first RC is out there, we'll be eagerly awaiting your [feedback](https://github.com/huggingface/transformers/issues/40822).