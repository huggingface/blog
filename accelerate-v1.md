---
title: "Accelerate 1.0, a staple across the 🤗 ecosystem"
thumbnail: /blog/assets/20_accelerate_library/accelerate_diff.png
authors:
- user: muellerzr
- user: marcsun13
- user: BenjaminB
---

# Accelerate 1.0, a staple across the 🤗 ecosystem

## What is Accelerate today?

3.5 years ago, Accelerate was a simple framework aimed at making training on multi-GPU and TPU systems easier
by having a low-level abstraction that simplified a *raw* PyTorch training loop:

![Sylvain's tweet announcing accelerate](https://raw.githubusercontent.com/muellerzr/presentations/master/talks/ai_dev_2024/sylvain_tweet.JPG)

Since then, Accelerate has expanded into a multi-faceted library aimed at tackling many of the common problems with both
large-scale training and large models in an age where 405 billion parameters are the new language model size. This involves:

* A flexible low-level training API, allowing for training on 6 different hardware accelerators (CPU, GPU, TPU, XPU, NPU, MLU) while maintaining 99% of your original training loop
* An easy-to-use command-line interface aimed at configuring and running scripts across different hardware configurations
* The birthplace of `device_map="auto"` and Big Model Inference, allowing users to not only perform inference on large models but now also aiding in training large models on small compute through techniques like parameter-efficient fine-tuning

These three facets have allowed Accelerate to become the foundation of **nearly every framework at Hugging Face**!

As the framework has been stable for nearly a year, we're excited to announce that as of today we've published **the first release candidates for Accelerate 1.0.0**!

This blog will detail:

1. Why did we decide to do 1.0?
2. What is the future for Accelerate, and where do we see PyTorch as a whole going?
3. What are the breaking changes and deprecations that occurred, and how can you migrate over easily?

## Why 1.0?

The plans for releasing 1.0.0 have been in the works for over a year now. The API has been roughly at a point where we wanted,
centering on the `Accelerator` side simplifying much of the configuration and making it more extensible. However, we knew
there were a few missing pieces before we could call the "base" of `Accelerate` "feature complete":

* Integrating FP8 support of both MS-AMP and `TransformersEngine` (read more [here](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/transformer_engine) and [here](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8/ms_amp))
* Supporting orchestration of multiple models when using DeepSpeed ([Experimental](https://huggingface.co/docs/accelerate/usage_guides/deepspeed_multiple_model))
* `torch.compile` support for the big model inference API
* Integrating `torch.distributed.pipelining` as an [alternative distributed inference mechanic](https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference#memory-efficient-pipeline-parallelism-experimental)
* Integrating `torchdata.StatefulDataLoader` as an [alternative dataloader mechanic](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/checkpointing.py)

As of this release, we can now say that these items are supported in the framework and their APIs are relatively stable.

By flipping the switch to 1.0, a safe and stable fallback for users, we can be prepared for some potentially radical
rewrites and changes in the codebase as new technologies come to fruition.

## The future of Accelerate

As previously mentioned, we're going through with this change because we foresee some radical changes in the PyTorch ecosystem very soon:

* As part of the multiple-model DeepSpeed support, we found that while generally how DeepSpeed is currently *could* work, some heavy changes to the overall API may eventually be needed as we work to support simple wrappings to prepare models for any multiple-model training scenario.
* With [torchao](https://github.com/pytorch/ao) and [torchtitan](https://github.com/pytorch/torchtitan) picking up steam, they hint at the future of PyTorch as a whole. Aiming at more native support for FP8 training, a new distributed sharding API, and support for a new version of FSDP, FSDPv2, we predict that much of the internals and general usage API of Accelerate will need to change (hopefully not too drastic) to meet these needs as the frameworks slowly become more stable.
* Riding on `torchao`/FP8, many new frameworks are bringing in different ideas and implementations on how to make FP8 training work and be stable (`transformer_engine`, `torchao`, `MS-AMP`, `nanotron`, to name a few). Our aim with Accelerate is to house each of these implementations in one place with easy configurations, to let users explore and test out each one as they please, intending to find the ones that wind up being the most stable and flexible in the end. It's a rapidly accelerating (no pun intended) field of research, especially with NVIDIA's FP4 training support on the way, and we want to make sure that not only can we support each of these methods, but aim to provide **solid benchmarks for each** to show their tendencies out-of-the-box (with minimal tweaking) compared to native BF16 training

We're extremely excited about the future of distributed training in the PyTorch ecosystem, and we want to make sure that Accelerate is there every step of the way providing a lower barrier to entry for these new techniques. By doing so, we hope the community will continue to experiment and learn together as we find the best techniques for training and scaling larger models on more complex computing systems.

## How to try it out

To try the first release candidate for Accelerate today, please use one of the following methods:

* pip:

```bash
pip install git+https://github.com/huggingface/accelerate@v1.0.0.rc0
```

* Docker:

```bash
docker pull huggingface/accelerate:gpu-release-1.0.0.rc
```

Valid release tags are:
* `gpu-release-1.0.0.rc`
* `cpu-release-1.0.0.rc`
* `gpu-fp8-transformerengine-release-1.0.0.rc`
* `gpu-deepspeed-release-1.0.0.rc`

## Migration assistance

Below are the full details for all deprecations that are being enacted as part of this release:

* Passing in `dispatch_batches`, `split_batches`, `even_batches`, and `use_seedable_sampler` to the `Accelerator()` should now be handled by creating an `accelerate.utils.DataLoaderConfiguration()` and passing this to the `Accelerator()` instead (`Accelerator(dataloader_config=DataLoaderConfiguration(...))`)
* `Accelerator().use_fp16` and `AcceleratorState().use_fp16` have been removed, this should be replaced by checking `accelerator.mixed_precision == "fp16"`
* `Accelerator().autocast()` no longer accepts a `cache_enabled` argument. Instead, an `AutocastKwargs()` instance should be used which handles this flag (among others)
* `accelerate.utils.is_tpu_available` should be replaced with `accelerate.utils.is_torch_xla_available`
* `accelerate.utils.modeling.shard_checkpoint` should be replaced with `split_torch_state_dict_into_shards` from the `huggingface_hub` library
* `accelerate.tqdm.tqdm()` no longer accepts `True`/`False` as the first argument, and instead `main_process_only` should be passed in as a named argument
* `ACCELERATE_DISABLE_RICH` is no longer a valid environmental variable, and instead one should manually enable `rich` traceback by setting `ACCELERATE_ENABLE_RICH=1`
* The FSDP setting `fsdp_backward_prefetch_policy` has been replaced with `fsdp_backward_prefetch`

## Closing thoughts

Thank you so much for using Accelerate, it's been amazing watching a small idea turn into over 100 million downloads and nearly 300,000 **daily** downloads over the last few years.

As users test out the release candidates, we will slowly roll out a few more versions before finally pushing the 1.0 release to give you all enough time to fully migrate and resolve any lingering deprecations.

Please stay tuned for more information!