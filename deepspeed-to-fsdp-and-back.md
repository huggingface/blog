---
title: "From DeepSpeed to FSDP and Back Again with Hugging Face Accelerate"
authors:
- user: mirinflim
- user: muellerzr
- user: stas00
---

# From DeepSpeed to FSDP and Back Again With Hugging Face Accelerate: How Interchangeable Are They?

There are two popular implementations of the [ZeRO Redundancy Optimizer (Zero)](https://arxiv.org/abs/1910.02054) algorithm in the community, one from [DeepSpeed](https://github.com/microsoft/DeepSpeed) and the other from [PyTorch](https://pytorch.org/docs/stable/fsdp.html). Hugging Face [Accelerate](https://huggingface.co/docs/accelerate/en/index) exposes both these frameworks for the end users to train/tune their models. This blog highlights the differences between how these backends are exposed through Accelerate. To enable users to seamlessly switch between these backends, we [upstreamed a precision related change](https://github.com/huggingface/accelerate/issues/2624) and a [concept guide](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed).

## Are FSDP and DeepSpeed Interchangeable?

Recently we ported a training pipeline from DeepSpeed to FSDP. We were surprised when
the results obtained differed *substantially*. The specific model used was a Mistral-7B loaded in
half-precision (`bfloat16`). While the DeepSpeed (blue) loss had converged well, the FSDP (orange)
loss was not decreasing, as can be seen in Figure 1.

Initially we thought that FSDP may be computing the mean of the gradients rather than the sum,
hence requiring a higher learning rate. So, we bumped up the learning rate by 4x since we were
using 4 GPUs. Then we saw the following loss behavior, shown in Figure 2:

[[ZACH: INSERT FIGURE TWO HERE]]

We hypothesized that the learning rate may need scaling by the number of GPUs and bumped up the learning rate by 4x since we were using 4 GPUs. Then, we saw the following loss behavior, shown in Figure 2. 

It looked like the desired behavior had been achieved by scaling the FSDP learning rate by the number of GPUs! However,  when we tried a different learning rate (`1e-5`) without scaling, we observed similar loss and gradient norm characteristics for both frameworks, shown in Figure 3.
## The Investigation Unfolds: Precision Matters

Inside the `DeepSpeed` codebase, specifically, in the implementation of
`DeepSpeedZeroOptimizer_Stage3` (as the name implies, what handles doing Stage 3 optimizer sharding), we noticed that the `trainable_param_groups`, the parameter groups being trained on, pass through an 
internal `_setup_for_real_optimizer` function call, which calls another function called `_create_fp32_partitions`.

This function was upcasting from `bf16` to `fp32` (full precision) internally, and it always kept the master weights in `fp32` *by design*. This upcasting meant that the optimizer could converge at learning rates that struggled in lower precisions. In Figure 1, FSDP was not converging with `1e-6`, and it was just a coincidence that bumping up the learning rate to `4e-6` (4x) just happened to be sufficient for converging, as shown in Figure 2. 

The crux of the issue is how these two frameworks handle training on and utilizing lower precisions.

During distributed training, before the model and optimizer parameters are split across GPUs they are first "flattened" to
a one-dimensional `torch.Tensor` (or, a list). FSDP and DeepSpeed use different `dtype`s for these "flattened"
parameters which has ramifications for PyTorch Optimizers. Table 1 outlines the processes for both
frameworks. The "Local" column indicates if the process occurs per-GPU, and therefore the memory
overhead from upcasting is amortized by the number of GPUs.

| **Process**                                                                              | **Local?** | **Framework**     | **Details**                                                                                                                                                                |
| ---------------------------------------------------------------------------------------- | ---------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loading the model in (such as `AutoModel.from_pretrained(..., torch_dtype=torch_dtype)`) | ❌         |                   |                                                                                                                                                                            |
| Preparation, such as creation of the "flattened parameters"                              | ✅         | FSDP<br>DeepSpeed | utilizes `torch_dtype`<br>disregards `torch_dtype` and is created in `float32`                                                                                             |
| Optimizer initialization                                                                 | ✅         | FSDP<br>DeepSpeed | creates parameters in `torch_dtype`<br>creates parameters in `float32`                                                                                                     |
| Training Step (forward, backward, reduction)                                             | ❌         | FSDP<br>DeepSpeed | follows [fsdp.MixedPrecision](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision)<br>follows `deepspeed_config_file` mixed precision settings |
| Optimizer (pre-step)                                                                     | ✅         | FSDP<br>DeepSpeed | upcasting (if any) to `torch_dtype`<br>upcasting everything to `float32`                                                                                                   |
| Optimizer (actual step)                                                                  | ✅         | FSDP<br>DeepSpeed | occurs in `torch_dtype`<br>occurs in `float32`                                                                                                                             |

> Table 1: Summary of how FSDP and DeepSpeed handle mixed precision

A few takeaways:
* As noted in the [🤗 Accelerate Documentation], a rule of thumb when performing mixed precision
is to keep trainable parameters in `torch.float32`.
* The upcasting done by `DeepSpeed` may have negligible effect on memory
consumption when sharding over a large number of GPUs, however when using
a small number of GPUs the 2x increase in memory
consumption can be **significant**.
* The torch-native implementation of FSDP does not force upcasting, allowing a
user to operate `torch.Optimizer` in low precision. This offers more flexibility than
the native upcasting of `DeepSpeed`.


## Harmonizing DeepSpeed and FSDP in 🤗 Accelerate

To better align DeepSpeed and FSDP in 🤗 Accelerate, [Stas Bekman](https://github.com/stas00) suggested performing the upcasting automatically
for FSDP when mixed precision was enabled. This was completed in [this accelerate PR](https://github.com/huggingface/accelerate/pull/2674) and has been
available since the 0.30.0 release.

[[ZACH: INSERT FIGURE FOUR HERE]]

The result of this PR is to allow FSDP to operate in two modes:
-  A “mixed-precision” mode like the DeepSpeed counterpart
- A low precision mode for memory constrained scenarios, as shown in Figure 5.

The two new FSDP modes are summarized in Figure 6 and compared with DeepSpeed.
| **Framework**             | **Model Loading (`torch_dtype`)** | **Mixed Precision** | **Preparation (Local)** | **Training** | **Optimizer (Local)** |
| ------------------------- | --------------------------------- | ------------------- | ----------------------- | ------------ | --------------------- |
| FSDP (memory-constrained) | `bf16`                            | default (none)      | `bf16`                  | `bf16`       | `bf16`                |
| FSDP (aligned mode)       | `bf16`                            | `bf16`              | `fp32`                  | `bf16`       | `fp32`                |
| DeepSpeed                 | `bf16`                            | `bf16`              | `fp32`                  | `bf16`       | `fp32`                |

> Table 2: Summary of the two new FSDP modes and comparisons with DeepSpeed

## Throughput results

We use the [IBM Granite 7B](https://huggingface.co/ibm-granite/granite-7b-base) model (which follows the Meta Llama2 architecture) for throughput comparisons. We compare Model Flops Utilization (MFU) and tokens/sec/GPU metrics and show them for FSDP (full sharding) and DeepSpeed (Zero3). 

We used four A100 GPUs as before with the following hyperparameters:
- Batch size of 8
- Model loaded in `torch.bfloat16`
- Mixed precision is the same dtype. 

Table 3 shows that FSDP and DeepSpeed are expected to perform similarly. 

> We intend to follow up with a comprehensive throughput comparison and approaches to improve throughput (e.g., 4D masks with packing, torch.compile, selective activation checkpointing) as large scale alignment techniques like [InstructLab](https://github.com/instructlab) and [GLAN](https://arxiv.org/abs/2402.13064) become popular.

| **Framework**       | **Tokens / sec / device** | **Step time (s)** | **Model Flops Utilization (MFU)** |
| ------------------- | ------------------------- | ----------------- | --------------------------------- |
| FSDP (aligned mode) | 3158.7                    | 10.4              | 0.41                              |
| DeepSpeed           | 3094.5                    | 10.6              | 0.40                              |

> Table 3: Ballpark throughput comparisons between FSDP and DeepSpeed on four A100 GPUs.

## Closing thoughts

We provided a [new concept guide](https://huggingface.co/docs/accelerate/v0.31.0/en/concept_guides/fsdp_and_deepspeed) to help users migrate between the two frameworks. The guide helps users answer questions such as:
* How do we achieve equivalent sharding strategies?
* How do we perform efficient model loading?
* How is weight prefetching managed in FSDP and DeepSpeed?
* What is the equivalent of FSDP wrapping in DeepSpeed?

We consider various modes of configuring these frameworks in 🤗 Accelerate,
- From the command line during `accelerate launch`
- From the various `Plugin` classes 🤗 Accelerate provides for (`DeepSpeed`)[https://huggingface.co/docs/accelerate/main/en/package_reference/deepspeed] and (`FSDP`)[https://huggingface.co/docs/accelerate/main/en/package_reference/fsdp]

🤗 Accelerate makes it almost **trivial** to switch between FSDP and DeepSpeed, with the majority of it being an Accelerate config file change (see the new concept guide for instructions on this).

Besides the config
change, some of the other considerations (also outlined in the guide) are differences in how checkpoints are handled, etc.

All experiments in this blog can be reproduced with the code from the [original 🤗 Accelerate issue](https://github.com/huggingface/accelerate/issues/2624).

We intend to follow up with throughput comparisons at scale and techniques to better utilize those GPUs for tuning and alignment jobs while maintaining model quality.
## Acknowledgements

This is an effort that involved several teams across multiple organizations to come together. It started at IBM Research, specifically Aldo Pareja who found the issue and Fabian Lim who identified the precision gaps and fixed this issue. Zach Mueller and [Stas Bekman](https://github.com/stas00) have been phenomenal in providing feedback and the fixes to accelerate. Less Wright from the PyTorch Team at Meta was very helpful in questions on FSDP parameters. Finally, we would also like to thank the [DeepSpeed team](https://www.deepspeed.ai/) for providing feedback on this blog.
