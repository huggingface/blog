---
title: Fast Stable Diffusion on your Mac using Swift Diffusers
thumbnail: /blog/assets/fast-mac-diffusion/thumbnail.png
authors:
- user: pcuenq
- user: reach-vb
---

# Fast Stable Diffusion with Diffusers for Mac

<!-- {blog_metadata} -->
<!-- {authors} -->

Diffusers for Mac is a native app to generate images from a text description of what you want. It uses state-of-the-art diffusion models, like Stable Diffusion, contributed by the community to the Hugging Face Hub, and converted to Core ML for maximum performance. We have just released version 1.1 of the app in the Mac App Store, with significant performance improvements, a lot of UI tweaks and some bug fixes. We think it's a solid foundation on which to build new features. And it's [fully open source](https://github.com/huggingface/swift-coreml-diffusers) with a [permissive license](https://github.com/huggingface/swift-coreml-diffusers/blob/main/LICENSE), so you can build on it too!

**TODO**: screenshot of the new UI

## What exactly is Diffusers for Mac anyway?

The Diffusers app ([App Store](https://apps.apple.com/app/diffusers/id1666309574), [source code](https://github.com/huggingface/swift-coreml-diffusers)) is the Mac counterpart to our [`diffusers` 🧨 library](https://github.com/huggingface/diffusers). This library is written in Python with PyTorch, and uses a modular design to train and run diffusion models. `diffusers` supports many different models and tasks, is highly configurable and very well optimized. It runs on Mac, too, using PyTorch's [`mps` accelerator](https://huggingface.co/docs/diffusers/optimization/mps), which is an alternative to `cuda` on Apple Silicon.

Why would you want to run a native Mac app then? There are many reasons:
- It uses Core ML models, instead of the original PyTorch ones. This is important because they allow for [additional optimizations](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon) relevant to the specifics of Apple hardware, and because Core ML models can run on all the compute devices in your system: the CPU, the GPU and the Neural Engine, _at once_ – the Core ML framework will decide what portions of your model to run on each device to make it as fast as possible. PyTorch's `mps` device cannot use the Neural Engine.
- It's a Mac app! We try to follow Apple's design language and guidelines so it feels at home in your Mac. No need to use the command line, create virtual environments or fix dependencies.
- It's local, and private. You don't need credits for online services and won't experience long queues – just generate all the images you want and use them for fun or work. Privacy is guaranteed: your prompts and images are yours to use, and will never leave your computer (unless you choose to share them).
- [It's open source](https://github.com/huggingface/swift-coreml-diffusers), and it uses Swift, Swift UI and the latest languages and technologies for Mac and iOS development. If you are technically inclined, you can use Xcode to extend the code as you like. We welcome your contributions, too! **TODO**: Link to "good first issues".

## Performance Benchmarks

**TL;DR:** Generation is up to **twice as fast** on Diffusers 1.1, depending on the computer you use. Benchmark and details follow.

The first version of the app we released a few days ago runs models on a combination of CPU and GPU, and does not use the Neural Engine. This is because it produced very good results on our test systems (i.e., my iMac and my laptop), it's most compatible, and requires less downloads. Use of the Neural Engine, or ANE, requires a different version of the Core ML model.

Since we released we've done a lot more testing on our personal systems, and on additional Macs we rented from cloud services. These are the results of our benchmark:

**TODO** Table.

The amount of memory does not seem to play a big factor on performance, but the CPU performance (i.e., the number of _performance cores_ in the system) does. My M1 Max laptop is a lot faster using the GPU than it is with the ANE. However, standard M1 processors found in Mac Minis are **twice as fast** using ANE than GPU. Interestingly, using _both_ the GPU and ANE accelerators does not improve performance with respect to the best results obtained with just one of them.

So for version 1.1 we are now automatically selecting the best accelerator based on the computer where the app runs. However, we don't have access to many other Mac models because they are not offered by any cloud services. For example, we haven't been able to test performance on any of the "Pro" Macbook Pro variants. If you'd like to help us gather data to keep improving the app, read on!

## Community Call for Benchmark Data

We are interested to run performance benchmarks on Mac models we don't have access to. If you'd like to help, we've created [this GitHub issue](todo: create with instructions and a results template) where you can post your results. We'll use them to optimize performance on an upcoming version of the app. We are particularly interested on M1 Pro, M2 Pro and M2 Mac architectures :)

**TODO**: screenshot with a crop of the compute units selector.

## Other Improvements in Version 1.1

In addition to the performance optimization and fixing a few bugs, we have focused on adding new features while trying to keep the UI as simple and clean as possible. Most of them are obvious (guidance scale, optionally disable the safety checker, allow generations to be canceled). My favorite ones are the model download indicators, and a shortcut to reuse the seed from a previous generation in order to tweak the generation parameters:

**TODO** gif

Version 1.1 also includes additional information about what the different generation settings do. We want Diffusers for Mac to make image generation as approachable as possible to all Mac users, not just technologists.

## Next Steps

We believe there's a lot of untapped potential for image generation in the Apple ecosystem. In future updates we want to focus on the following:

- Easy access to additional models from the Hub. Run any Dreambooth or fine-tuned model from the app, in a Mac-like way.
- Release a version for iOS and iPadOS.

There are many more ideas that we are considering. If you'd like to suggest your own, you are most welcome to do so [in our GitHub repo](https://github.com/huggingface/swift-coreml-diffusers).
