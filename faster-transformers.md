---
title: "Updates in transformers that enabled gpt-oss" 
thumbnail: /blog/assets/faster-transformers/thumbnail.png
authors:
- user: ariG23498
- user: sergiopaniego
- user: reach-vb
- user: pcuenq
- user: ArthurZ
- user: SaylorTwift
- user: cyrilvallez
---

# Updates in transformers that enabled gpt-oss

OpenAI recently released their [GPT-OSS series of models](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4). The models feature some novel techniques like MXFP4 quantization, efficient kernels, a brand new chat format, and more. To enable the release of gpt-oss through `transformers`, we have upgraded the [library](https://github.com/huggingface/transformers/) considerably. The updates make it very efficient to **load**, **run**, and **fine-tune** the models.

In this blog post, we talk about all the upgrades in-depth. New features are usually motivated by innovative techniques from new models. We incorporate them as part of the `transformers` toolkit, so other models (current and future) can benefit from them too. Providing clean implementations of new methods allows the community to quickly understand and adopt them. Frameworks such as [`MLX`](https://github.com/ml-explore/mlx-lm/pull/354), [`llama.cpp`](https://github.com/ggml-org/llama.cpp/discussions/15396) or [`vLLM`](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) can use the `transformers` code as a reference to build their own implementations.

For this release, we worked on:

- [Zero-build Kernels, downloadable from the Hub](#zero-build-kernels-downloadable-from-the-hub)
- [MXFP4 Quantization](#mxfp4-quantization)
- [Tensor Parallelism](#tensor-parallelism)
- [Expert Parallelism](#expert-parallelism)
- [Dynamic Sliding Window Layer & Cache](#dynamic-sliding-window-layer--cache)
- [Continuous Batching & Paged Attention](#continuous-batching--paged-attention)
- [Load larger models faster](#load-larger-models-faster)

> [!NOTE]
> Best part: Most of these features should work across all major models within `transformers`!

## Zero-build Kernels, downloadable from the Hub

A kernel is a ***specialized***, compact program that runs on accelerators to execute tasks like matrix multiplications, activations, or normalizations. In eager PyTorch, operations trigger individual kernels sequentially, which is straightforward but inefficient due to memory transfers and launch overheads. PyTorch 2.0's `torch.compile` with backends like `TorchInductor` addresses this by automatically fusing and optimizing kernels, delivering `2–10× ` performance gains.

In addition, the community has created custom kernels for frequent combinations of operations *not just individual PyTorch ops like matmul*. For example, Flash Attention was created to optimize the critical attention block that defines the transformers architecture, and is present in many models including most LLMs. By carefully combining all the attention operations inside a single kernel, memory transfers are minimized, memory use is reduced, and speedups can be achieved.

The problem is that all these various kernels are available in separate libraries, which creates a dependency bloat if they were to be added to the transformers library. Furthermore, these kernels are not just Python code, they consist of low-level cuda code, glued together with C++ and exposed through a Python layer. This means they have to be compiled in the target system, which in turn requires whatever build system is required by each kernel library.
The [kernels package](https://huggingface.co/blog/hello-hf-kernels) solves this problem by downloading pre-built binaries of supported kernels from the Hub. You just indicate the kernel you want to use, and `kernels` will look for a version compatible with your system and download it on first use.

### Custom Kernels for GPT-OSS

[GPT-OSS](https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/src/transformers/models/gpt_oss/modeling_gpt_oss.py), a Mixture of Experts (MoE) model, is a big user of Kernels from the Hub. It leverages the following customized kernels:

1. `@use_kernel_forward_from_hub("RMSNorm")`
2. `@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`
3. MXFP4 triton kernels (covered [later](#mxfp4-in-transformers))

Behind the scenes, the decorators (1 and 2) simply point to community-contributed kernels. For example, `RMSNorm` comes from [`liger_kernels`](https://huggingface.co/kernels-community/liger_kernels), while the `MegaBlocksMoeMLP` kernel comes from [`megablocks`](https://huggingface.co/kernels-community/megablocks). Depending on your device (CUDA or ROCm) and whether you’re training or running inference, the right kernel is pulled in automatically.

This design is both **specific and general**: the MoE kernel is tailored to GPT-OSS, but the RMSNorm liger kernels is already being reused across multiple models. Because `kernels` pulls code from the Hub, you have to opt-in to this feature by passing `use_kernels=True` in your model instantiation, as shown below. We also enable `INFO` logging so you can easily verify that downloadable kernels are in use.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging
logging.basicConfig(level=logging.INFO)

model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    use_kernels=True,
)
```

Running a quick generation yields log messages like

```shell
INFO:root:Using layer `LigerRMSNorm` from repo `kernels-community/liger_kernels`
INFO:root:Using layer `MegaBlocksMoeMLP` from repo `kernels-community/megablocks`
```

**Figure 1** shows that, in the system we tested, these kernels work best for larger batch sizes. We always recommend to benchmark any performance-related changes as closely to your production conditions as possible.

| ![benchmark with and without kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/benchmark-kernels-with-without.png) |
| :--: |
| Figure 1: Benchmarking results of custom kernels |

> [!NOTE]
> You can explore and play with the benchmarking script [here](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/benchmark-kernels-with-without.py)

## MXFP4 Quantization

### Why quantize at all

Large language models are memory-hungry. Quantization reduces memory footprint by storing weights (and sometimes activations) in lower-precision formats. For reference, `FP32` uses 32 bits per number and `BF16` uses 16. By reducing bit width, we trade some precision for smaller models and faster memory movement.

If you want a visual primer on quantization trade-offs, [Maarten Grootendorst’s](https://huggingface.co/MaartenGr) article is excellent: [*A Visual Guide to Quantization*](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization).

### What is MXFP4

| ![explanation of mxfp4 format](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/mxfp4.png) |
| :--: |
| Figure 2: The E2M1 format used in the MXFP4 format |

`MXFP4` is a 4-bit floating format with E2M1 layout: 1 sign bit, 2 exponent bits, and 1 mantissa bit, as shown in **Figure 2**. On its own, E2M1 is very coarse. MXFP4 compensates with **blockwise scaling**:

- Vectors are grouped into blocks of 32 elements.
- Each block stores a shared scale that restores dynamic range when dequantizing.
- Inside each block, 4-bit values represent numbers relative to that scale.

This blockwise scheme lets `MXFP4` keep range while using very few bits. In practice, GPT-OSS 20B fits in roughly `16 GB` of VRAM and GPT-OSS 120B fits in roughly `80 GB` when `MXFP4` is active, which is the difference between “cannot load” and “can run on a single GPU.” The catch is that matrix multiplies now have to respect block scales. Doing this efficiently at scale requires dedicated kernels.

### MXFP4 in `transformers`

`transformers` now includes native support for MXFP4, leveraging optimized `triton` (MXFP4) kernels for enhanced performance. This builds on the community-driven kernel distribution [discussed earlier](#zero-build-kernels-downloadable-from-the-hub), utilizing pre-compiled kernels from the Hub to simplify deployment.

Key implementation details:

- Quantizer logic: Found in the [MXFP4 quantizer file](https://github.com/huggingface/transformers/blob/0997c2f2ab08c32c8e2f90aaad06e29a7108535b/src/transformers/quantizers/quantizer_mxfp4.py), this handles the core quantization process for MXFP4.
- Integration hooks: The [MXFP4 integration file](https://github.com/huggingface/transformers/blob/0997c2f2ab08c32c8e2f90aaad06e29a7108535b/src/transformers/integrations/mxfp4.py) enables seamless use of MXFP4 within the transformers framework.

To check if a model supports `MXFP4`, inspect its configuration:
```py
from transformers import GptOssConfig

model_id = "openai/gpt-oss-120b"
cfg = GptOssConfig.from_pretrained(model_id)
print(cfg.quantization_config)

# Example output:
# {
#   'modules_to_not_convert': [
#     'model.layers.*.self_attn',
#     'model.layers.*.mlp.router',
#     'model.embed_tokens',
#     'lm_head'
#   ],
#   'quant_method': 'mxfp4'
# }
```

If `'quant_method': 'mxfp4'` is present, the model will automatically use the MXFP4 pathway with Triton kernels when supported.

> [!NOTE]
> Thanks to this [pull request](https://github.com/huggingface/transformers/pull/40176), you can fine-tune gpt-oss models and save them directly to the Hub in MXFP4 format, streamlining deployment with optimized performance.

### Requirements and fallbacks

To run `MXFP4` on GPU you need:

1. `accelerate`, `kernels`, and `triton>=3.4` installed. Note that `Pytorch 2.8` already comes with `triton 3.4`, so you only need to manually install triton if using `Pytorch 2.7`.
2. NVIDIA GPU with compute capability `≥ 7.5`. This goes all the way back to Tesla, so you can run `gpt-oss-20b` on the free tiers of Google Colab and Kaggle, and on many consumer GPUs.

If these constraints are not met, `transformers` falls back to a higher-precision path (`BF16` is used by default), which requires about 4× the memory of MXFP4.

The [snippet](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/memory-requirements-quantized-vs-dequantized.py) loads GPT-OSS twice on CUDA: once with `Mxfp4Config(dequantize=True)` (memory intensive) and once in the default quantized path (memory efficient). **Figure 3** shows the amount of used VRAM after each load so you can visualize the savings.

| ![memory used with quantized vs dequantized models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/quantization.png) |
| :--: |
| Figure 3: Memory requirements for the quantized and dequantized models |

### Kernels for MXFP4

Efficient `MXFP4` requires kernels that understand 32-element blocks and their scales during GEMMs and fused ops. This is where **Kernels from the Hub** comes in again. `transformers` automatically pulls in the `MXFP4`-aware
tTriton kernels from the community repository when you load a model that needs them. The repository will appear in your local cache and will be used during the forward pass. For the `MXFP4` kernels one does not need to use the `use_kernels=True` parameter like before, it is set to default in `transformers`.

Quick sanity check with the Hugging Face cache CLI,  after running `gpt-oss-20b` on a GPU compatible with the triton MXFP4 kernels:

```shell
hf cache scan
```

Sample output:

```shell
REPO ID                          REPO TYPE SIZE ON DISK
-------------------------------- --------- ------------
kernels-community/triton_kernels model           536.2K
openai/gpt-oss-20b               model            13.8G
```

This indicates the MXFP4 kernels were fetched and are available for execution.

Let's run some benchmarks and see how well the MXFP4 kernels perform. In **Figure 4**, we see that the `MXFP4` kernels are even better than the custom MoE and RMSNorm kernels for larger batches.

| ![benchmark mxfp4 kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/benchmark-mxfp4.png) |
| :--: |
| Figure 4: MXFP4 kernel benchmark |

> [!NOTE]
> You can explore and play with the benchmarking script [here](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/benchmark-mxfp4-kernels.py)

## Tensor Parallelism

| ![explaining tensor parallelism](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/TP.png) |
| :--: |
| Figure 5: Explanation of tensor parallelism. |

Tensor Parallelism (TP) splits **tensors inside a layer** across multiple GPUs (as shown in **Figure 5**). Each GPU multiplies its shard in parallel, and then partial results are collected using all-gather or all-reduce operations.
This reduces per-GPU memory and keeps all GPUs working on the **same layer**, which improves throughput as sequence length or batch size grow. TP is communication-intensive and generally works best on a **single machine with fast intra-node links**.

### What this enables in `transformers`

`transformers` implements TP directly in `from_pretrained`. You can start with the predefined plan:

```python
# run with: torchrun --nproc-per-node 4 tp_gpt_oss.py
import torch
from transformers import PreTrainedTokenizerFast, GptOssForCausalLM

model_id = "openai/gpt-oss-120b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto", # built in TP support
    dtype="auto",
).eval()

messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain KV caching briefly."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

with torch.inference_mode():
    generations = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(generations[0][inputs["input_ids"].shape[-1]:]))
```

If you don’t have the infrastructure to run the above, you can just spawn a process on our GPUs using [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)!

```bash
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/tp_gpt_oss.py && \
  torchrun --nproc-per-node=4 tp_gpt_oss.py"
```

> [!NOTE]
> [`hf jobs`](https://huggingface.co/docs/huggingface_hub/guides/jobs) is available for all Hugging Face PRO & Enterprise users.

Under the hood, `tp_plan="auto"` selects a predefined sharding recipe for each layer and wires the necessary collectives. You can inspect the active plan with `print(model._tp_plan)` if you want to verify what is being sharded.

### When to reach for TP

Use TP when the model is too large for one GPU and you want **parallel compute**, not only memory placement. TP tends to scale throughput with more GPUs, especially for long sequences or larger batches.

> [!NOTE]
> If you are curious about how TP differs from `device_map="auto"` (memory placement), this short [Stack Overflow answer](https://stackoverflow.com/questions/78852192/choose-available-gpu-devices-with-device-map) explains the distinction and when to use each.

If you want to know more about TP, here are two must-read resources:

- [`transformers` guide](https://huggingface.co/docs/transformers/en/perf_infer_gpu_multi): Tensor parallelism, supported models, plans, and extension points.
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism): background on TP and its relationship to other parallelism modes.

## Expert Parallelism

Expert Parallelism (EP) shards **experts inside MoE layers** across GPUs. Each token is routed to one or a few experts, so only those experts run their feed-forward pass. Since experts are independent MLPs, we can place different experts on different ranks and exchange only the hidden states for the routed tokens. This keeps the matrix multiplies intact on each rank and replaces tensor slicing with routing and collectives.

Run with multiple processes using `torchrun`. EP is enabled via the distributed configuration and works with GPT-OSS MoE layers out of the box in transformers.

```python
# run with: torchrun --nproc-per-node 4 ep_gpt_oss.py
import torch
from transformers import PreTrainedTokenizerFast, GptOssForCausalLM
from transformers.distributed import DistributedConfig

model_id = "openai/gpt-oss-120b"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
model = GptOssForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(enable_expert_parallel=True), # enabling EP
    dtype="auto",
).eval()

messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Explain KV caching briefly."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

with torch.inference_mode():
    generations = model.generate(**inputs, max_new_tokens=128)

print(tokenizer.decode(generations[0][inputs["input_ids"].shape[-1]:]))
```

Here is how you would run using `hf jobs`
```bash
hf jobs run --detach --flavor l4x4 ghcr.io/astral-sh/uv:debian /bin/bash -c \
  "uv venv .venv --python 3.12 && \
  source .venv/bin/activate && \
  uv pip install --upgrade torch numpy transformers accelerate triton kernels && \
  wget https://huggingface.co/datasets/ariG23498/distributed/raw/main/ep_gpt_oss.py && \
  torchrun --nproc-per-node=4 ep_gpt_oss.py"
```

> [!NOTE]
> When you enable Expert Parallelism, Tensor Parallelism gets activated by default. This means you enjoy the best of both worlds!

## Dynamic Sliding Window Layer & Cache

`transformers` now has a [**`DynamicSlidingWindowLayer`**](https://github.com/huggingface/transformers/blob/64ae6e6b1de2c6822a53be46aba9db68f75ec595/src/transformers/cache_utils.py#L165) and a *config‑aware* [`DynamicCache`](https://github.com/huggingface/transformers/blob/64ae6e6b1de2c6822a53be46aba9db68f75ec595/src/transformers/cache_utils.py#L959). If the model config declares sliding‑window or hybrid attention, the cache **stops growing past the window** for those layers; if you don’t pass the config, behavior stays as before (full, ever‑growing KV).

This provides us with:

- **Much lower KV‑cache memory** for models with sliding or hybrid attention (e.g. GPT‑OSS). Cache growth plateaus once the window is reached (e.g., 4K for Mistral; 128 for GPT‑OSS sliding layers), instead of scaling linearly with total generated tokens. ([GitHub](https://github.com/huggingface/transformers/pull/40039), [Transformers](https://huggingface.co/docs/transformers/en/model_doc/mistral))
- **Speed/latency wins** on long prompts/long generations: smaller KV tensors mean lighter attention reads/writes and less memory bandwidth pressure, especially after the window is hit. (This is the central motivation behind sliding‑window/hybrid LLMs.) ([AI21](https://www.ai21.com/blog/rise-of-hybrid-llms/), [vLLM Blog](https://blog.vllm.ai/2025/08/05/gpt-oss.html))

### How to use it

The optimized cache is set by default, that means **you don't have to make any changes** to your existing code. If you want to create the `DynamicCache` explicitly here is how you would do it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
	model_id,
	dtype="auto",
	device_map="auto",
).eval()

messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

cache = DynamicCache(config=model.config) # create the cache with the model's config

generated = model.generate(
	**inputs,
	max_new_tokens=500,
	past_key_values=cache
)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

**Figure 6** showcases how much of a difference it makes for us to use the Dynamic KV Cache with sliding window attention.

| ![sliding window cache](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/dynamic-cache.png) |
| :--: |
| Figure 6: The memory analysis of dynamic cache with sliding window attention |


## Continuous Batching & Paged Attention

A typical autoregressive generation process looks like **Figure 7**. You input the prefill tokens, and the model predicts each new token one after the other until it predicts the EOS (End of Sequence) token.

| ![prefilling](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/prefill-tokens.png) |
| :--: |
| Figure 7: Autoregressive token generation |

Let’s see what the generation process looks like when we pass a **batch** of inputs. In **Figure 8** you notice that some generations finish off early than the others. This mismatch of length underutilizes the GPUs.

| ![static batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/static-batching.png) |
| :--: |
| Figure 8: Static batching of sequences |

This type of batching sequences is called *static batching*. While this is simple and easy to understand, it inherently comes with inefficiencies. Only after each sentence is completely generated can we move on to the next batch.

To bypass this issue, we use **dynamic batching** (also known as *continuous batching*). Instead of waiting for all the generation to finish, we schedule incoming requests to the completed generations. That way, as soon as a generation in a batch is complete, we prefill the batch with the next request. The process looks like **Figure 9**.

| ![continuous batching](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/dynamic-batching.png) |
| :--: |
| Figure 9: Continuous Batching of sequences |

Transformers supports continuous batching with the `generate_batch` API. Here is an official [script](https://github.com/huggingface/transformers/blob/0f1b128d3359a26bd18be99c26d7f04fb3cba914/examples/pytorch/continuous_batching_simple.py) that runs CB end to end on `Qwen/Qwen3-4B-Instruct-2507`.

We have also performed a benchmark between Continuous Batching and Static Batching with 100 samples. In Figure 9, we note that CB is quite faster than SB.

| ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/cb-sb.png) |
| :--: |
| Figure 9: Continuous vs Static Batching Tokens/Second |

> [!NOTE]
> You can play around with the benchmark here: [SB](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/sb-bench.py), [CB](https://huggingface.co/datasets/ariG23498/faster-transformers-scripts/blob/main/cb-bench.py)

## Load larger models faster

When you load a large model into your GPU, PyTorch needs to **reserve GPU memory for each layer’s weights**. Each of these requests (per layer) takes time, and for multi-billion-parameter models it can mean **thousands of tiny memory allocations**, adding up to a long wait before the model is ready. Instead of asking the GPU for new memory every single time, it can **hold on to a big chunk once** and then hand out slices from it quickly.

PyTorch allocators can do exactly this. The catch is that the allocator only gets fast *after* you’ve given it some memory to work with. If you don’t “stock the pantry” first, you still end up doing many slow trips to the market. This PR (🎉 [#36380](https://github.com/huggingface/transformers/pull/36380)) taught `transformers` to **pre-stock the pantry** before it starts copying model weights.

It:
- Looks at the `device_map` (where each layer will live).
- **Pre-allocates a big enough block on each GPU**.
- Then, as layers are copied in, they just slot neatly into this pre-reserved space.

This results in speedups in practice. In **Figure 11** we show the loading times of dequantized gpt-oss-20b with and without the allocator.

| ![speedup of loading models](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/faster-transformers/load-big-models.png) |
| :--: |
| Figure 11: Loading time of gpt-oss-2b model |

You have to make no changes to your existing code, as this is default behaviour in `transformers`. If you use **`device_map="auto"`** or provide your own device map, your model will now load faster automatically. If you’re running with **Tensor Parallel (`tp_plan="auto"`) and `torchrun`** you also benefit from companion changes that make multi-GPU loading smarter.

## Conclusion

`transformers` moves quickly and it is community-first. The library evolves at the pace of the field because contributors shape it in the open.

That velocity enables day-zero integrations like the GPT-OSS series. As the stack becomes increasingly [PyTorch-first](https://x.com/LysandreJik/status/1933201171130593530), it sheds bloat and doubles down on the PyTorch paths that matter in practice. The result is a cleaner core that still unlocks new capabilities through community kernels, quantization, and parallelism plans, while also
[standardizing model definitions](https://huggingface.co/blog/transformers-model-definition) so that architectures supported in transformers seamlessly extend across the wider ecosystem.

The direction is constant: serve the needs of the community. This post is a snapshot meant to put the key ideas in one place, not a rolling changelog. It will not be updated often. For the latest details, check the [docs](https://huggingface.co/docs/transformers/index) and [release notes](https://github.com/huggingface/transformers/releases), and keep sharing feedback so the next steps reflect what you need.

## Read More

If you want to go further into particular topics, here is a list of links that one should visit:

1. https://github.com/huggingface/gpt-oss-recipes
2. https://huggingface.co/blog/welcome-openai-gpt-oss
3. https://cookbook.openai.com/topic/gpt-oss
4. https://huggingface.co/docs/transformers/v4.53.3/en/perf_infer_gpu_multi
5. https://x.com/carrigmat/status/1952779877569978797
6. https://www.youtube.com/watch?v=bbkcEiUjehk
7. https://github.com/huggingface/transformers/pull/36380
8. https://github.com/huggingface/transformers/pull/36335
9. https://github.com/huggingface/transformers/pull/40039
10. https://hanlab.mit.edu/blog/streamingllm