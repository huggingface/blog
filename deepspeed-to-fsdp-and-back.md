---
title: "From DeepSpeed to FSDP and Back Again with Hugging Face Accelerate"
authors:
- user: mirinflim
- user: aldopareja
- user: muellerzr
- user: stas00
---

# A Hugging Face Accelerate Story of Multiple Backends: FSDP and DeepSpeed

There are two popular implementations of the [ZeRO Redundancy Optimizer (Zero)](https://arxiv.org/abs/1910.02054) algorithm in the community, one from [DeepSpeed](https://github.com/microsoft/DeepSpeed) and the other from [PyTorch](https://pytorch.org/docs/stable/fsdp.html). Hugging Face [Accelerate](https://huggingface.co/docs/accelerate/en/index) exposes both these frameworks for the end users to train/tune their models. This blog highlights the differences between how these backends are exposed through Accelerate. To enable users to seamlessly switch between these backends, we [upstreamed a precision related change](https://github.com/huggingface/accelerate/issues/2624) and a [concept guide](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed).

## Are FSDP and DeepSpeed Interchangeable?

Recently we tried running a training pipeline with DeepSpeed and PyTorch FSDP. We noticed that the results obtained differed. The specific model was Mistral-7B base and it was loaded in half-precision (`bfloat16`). While the DeepSpeed (blue) loss had converged well, the FSDP (orange) loss was not decreasing, as can be seen in Figure 1.

![Figure 1](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/1666e2fac64a29bdb6993c3856cc2964db208be8640f5d9e23b4a2111c0c800c?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27figure_1.png%3B+filename%3D%22figure_1.png%22%3B&response-content-type=image%2Fpng&Expires=1718385931&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODM4NTkzMX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy8xNjY2ZTJmYWM2NGEyOWJkYjY5OTNjMzg1NmNjMjk2NGRiMjA4YmU4NjQwZjVkOWUyM2I0YTIxMTFjMGM4MDBjP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=hNbNoQ8MClNsqOVTlWLAZF-3MJ5xIYoue7BoImRSoZyV3tb0lqYYqssLsLRCFE95PSkMj%7Eq0Yl7CGAEwAdTovDgeze5G46vNCDdkS8f5IXNHk%7ExCKQ4M0nRV4RQK9tDYiQOxFjRnU8STKtZtV8SejnVC1EapBgiNyCuptOWfpKAmsPqVds4LoiVMfHfk4ZV1Q41O4HIsLl4F9Iwnl37kHeykNb0tZgaKU8JaIoGBuDmpRESc3pjuQ2OHZN4TktQf4yaXuprAN2iuojVPHyc8KRMf3JOldq9yL22dynIvg6EWsx8iZmKaNHQuCu7OEa-694iYl61wqaQNiexE9qkcOw__&Key-Pair-Id=KVTP0A1DKRTAX)

We hypothesized that the learning rate may need scaling by the number of GPUs and bumped up the learning rate by 4x since we were using 4 GPUs. Then, we saw the following loss behavior, shown in Figure 2. 

![Figure 2](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/ba57b663be4a9252e3a3bc3db02c992b1958a3fd72b2278198bd08fb3d6277c6?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27figure_2.png%3B+filename%3D%22figure_2.png%22%3B&response-content-type=image%2Fpng&Expires=1718385962&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODM4NTk2Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9iYTU3YjY2M2JlNGE5MjUyZTNhM2JjM2RiMDJjOTkyYjE5NThhM2ZkNzJiMjI3ODE5OGJkMDhmYjNkNjI3N2M2P3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=wdNZQPB2N5vQSBF0izLR8FK1y-DM10RyYpJnUUOgkG%7ExQ2aDRueltp6edXG-f6cln-c8BzaL-w0YUSyQkEiegkScCjAg46ttqhv-H5IRlvMJ2TvmunZXEtV3CKxShxlSyjaq7BOcR0VzlyIPFybMIcD9%7EDR6dHICDFO-mYzwKLuOTCKb7AU3GaEAupKjELm1XRsniDDFa4QpOA7zTnNOXDTR8uMVTx0itvhcASHKkcj%7EJ4hsdfAzVj1nxz0EHKvmudRaeC6o5CBw%7EwKm2At%7EHFN9Kgr7eZ48pze5f-rvLvQHn6BuQF-VwAN8UJCwNj4ilCzdhPpa7nw2VFmRQAjRrw__&Key-Pair-Id=KVTP0A1DKRTAX)

It looked like the desired behavior had been achieved by scaling the FSDP learning rate by the number of GPUs! However, when we tried a different learning rate (`1e-5`) without scaling, we observed similar loss and gradient norm characteristics for both frameworks, shown in Figure 3.

![Figure 3](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/e86c69648c3a6bfc8709c5149e28b29dbd5d61af621201175e97abbcf57a26ee?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27figure_3.png%3B+filename%3D%22figure_3.png%22%3B&response-content-type=image%2Fpng&Expires=1718385978&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODM4NTk3OH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9lODZjNjk2NDhjM2E2YmZjODcwOWM1MTQ5ZTI4YjI5ZGJkNWQ2MWFmNjIxMjAxMTc1ZTk3YWJiY2Y1N2EyNmVlP3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=r4pufHy2LN0gemtuivnUDVyaDiNkssKsfD-6K%7EGfao%7EoRwOS9IRe4AqiIZkyFIpnbs4yRZCrTdhIlfzxZyQzlnkia29CuEGIujiHo4uR3wTssI06GutEvaxDGzPnfkOiNqogr24BySHzEq1cBKytaiIzqlTbETcRkeI-FckCZ9a3wjnEEp%7EPd1oy1HYhAARlbPWJrzhBtuNgMsHjG7bA0WqfiuX-BuwYNVXuyfH2uWV4SKZ3pqz-P%7EAmmOSmvat3Yu2PZmtbe6grJGymMSqCZy%7Ej-RzwmrNFLNf14M9WA-P7MgYoB1dHDKosdj7CA9V67eevYIV3RiQSQvF37SYj9A__&Key-Pair-Id=KVTP0A1DKRTAX)


## Precision Matters

Inside the `DeepSpeed` codebase, specifically, in the implementation of
`DeepSpeedZeroOptimizer_Stage3` (as the name implies, what handles doing Stage 3 optimizer sharding), we noticed that the `trainable_param_groups`, the parameter groups being trained on, pass through an 
internal `_setup_for_real_optimizer` function call, which calls another function called `_create_fp32_partitions`.
As the `fp32` in the name suggests, `DeepSpeed` was performing upcasting internally, and it always keeps its master weights in `fp32` by design. This upcasting to full precision meant that the optimizer could converge at learning rates that it would not converge in lower precision. The earlier observations were artifacts of this precision difference.

In FSDP, before the model and optimizer parameters are distributed across GPUs, they are first "flattened" to a one-dimensional tensor. FSDP and DeepSpeed use different `dtype`s for these "flattened" parameters which has ramifications for PyTorch optimizers. Table 1 outlines the processes for both frameworks; the "Local" column indicates the process occurring per-GPU, therefore the memory overhead from upcasting is amortized by the number of GPUs.

| **Process**                                                                              | **Local?** | **Framework**     | **Details**                                                                                                                                                                |
| ---------------------------------------------------------------------------------------- | ---------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loading the model in (such as `AutoModel.from_pretrained(..., torch_dtype=torch_dtype)`) | ❌         |                   |                                                                                                                                                                            |
| Preparation, such as creation of the "flattened parameters"                              | ✅         | FSDP<br>DeepSpeed | utilizes `torch_dtype`<br>disregards `torch_dtype` and is created in `float32`                                                                                             |
| Optimizer initialization                                                                 | ✅         | FSDP<br>DeepSpeed | creates parameters in `torch_dtype`<br>creates parameters in `float32`                                                                                                     |
| Training Step (forward, backward, reduction)                                             | ❌         | FSDP<br>DeepSpeed | follows [fsdp.MixedPrecision](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.MixedPrecision)<br>follows `deepspeed_config_file` mixed precision settings |
| Optimizer (pre-step)                                                                     | ✅         | FSDP<br>DeepSpeed | upcasting (if any) to `torch_dtype`<br>upcasting everything to `float32`                                                                                                   |
| Optimizer (actual step)                                                                  | ✅         | FSDP<br>DeepSpeed | occurs in `torch_dtype`<br>occurs in `float32`                                                                                                                             |

> Table 1: Summary of how FSDP and DeepSpeed handle mixed precision

A few takeaway points:
* As noted in the 🤗 Accelerate documentation, a rule of thumb when performing mixed precision is to keep trainable parameters in torch.float32. 
* Upcasting, as is done in `DeepSpeed`, may have negligible effect on memory consumption when sharding over a large number of GPUs. However, when using `DeepSpeed` on a small number of GPUs, the 2x increase in memory consumption can be significant.
* The torch-native implementation of FSDP does not force upcasting, allowing a user to operate PyTorch optimizers in low precision. This offers more flexibility than the native upcasting of `DeepSpeed`.


## Harmonizing DeepSpeed and FSDP in 🤗 Accelerate

To better align DeepSpeed and FSDP in 🤗 Accelerate, we can perform upcasting automatically for FSDP when mixed precision is enabled. We created a pull request with this change that was included in the [0.30.0 release](https://github.com/huggingface/accelerate/releases/tag/v0.30.0).

![Figure 4](https://cdn-lfs.huggingface.co/datasets/huggingface/documentation-images/ec55b20f2ce9609efe736c5718840d15f1440f83ea9e8f45f29b6abb6d5251c6?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27figure_4.png%3B+filename%3D%22figure_4.png%22%3B&response-content-type=image%2Fpng&Expires=1718385992&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODM4NTk5Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9kYXRhc2V0cy9odWdnaW5nZmFjZS9kb2N1bWVudGF0aW9uLWltYWdlcy9lYzU1YjIwZjJjZTk2MDllZmU3MzZjNTcxODg0MGQxNWYxNDQwZjgzZWE5ZThmNDVmMjliNmFiYjZkNTI1MWM2P3Jlc3BvbnNlLWNvbnRlbnQtZGlzcG9zaXRpb249KiZyZXNwb25zZS1jb250ZW50LXR5cGU9KiJ9XX0_&Signature=Vt8AJKjQptx1x3bmzVzfWHbu3ZQY1xjQcrk6A36yHh2gZikqzZbSbK3j82JItRtn1FaxrlO5mK2oZAoGIxZGlIMqSFc%7EWQmr51wBTChSbgMMyup0WeBXtrCq8%7Eb5KPaD7MCHSYEc1CAJWwOwbUfjqqJFFtnj9J1Yh%7Ea%7EKFmrY-FDdu7fknw%7EjEBuIZE%7EPJS3nPBpVTe4fRt4vKgJnqUWbd9Cg0qny6b1bgPSsDWW5qjR0lGqWy%7EGXihEbBwuoV%7EhpCDMiv3K64P5iCQKX3VQVzsXNQfvWYssOqXYuijLgaNoT6r6BbYmnNiVkyvEHnwrTvEwvGk%7ECBm8%7E1kVr1m1jg__&Key-Pair-Id=KVTP0A1DKRTAX)

The result of this PR is to allow FSDP to operate in two modes:
- A “mixed-precision” mode like the DeepSpeed counterpart
- A low precision mode for memory constrained scenarios, as shown in Figure 4.

The two new FSDP modes are summarized in Table 2 and compared with DeepSpeed.
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

This is an effort that involved several teams across multiple organizations to come together. It started at IBM Research, specifically Aldo Pareja who found the issue and Fabian Lim who identified the precision gaps and fixed this issue. Zach Mueller and [Stas Bekman](https://github.com/stas00) have been phenomenal in providing feedback and the fixes to accelerate. Less Wright from the PyTorch Team at Meta was very helpful in questions on FSDP parameters. Finally, we would also like to thank the [DeepSpeed](https://www.deepspeed.ai/) team for providing feedback on this blog.
